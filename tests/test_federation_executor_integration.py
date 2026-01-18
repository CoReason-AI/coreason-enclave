# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_enclave

import json
import uuid
from typing import Any, Dict, Generator
from unittest.mock import MagicMock, patch

import pytest
import torch
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import ReturnCode, Shareable
from nvflare.apis.signal import Signal
from torch.utils.data import DataLoader, TensorDataset

from coreason_enclave.federation.executor import CoreasonExecutor
from coreason_enclave.schemas import AttestationReport


class TestCoreasonExecutorSecurity:
    @pytest.fixture
    def mock_job_config(self) -> Dict[str, Any]:
        return {
            "job_id": str(uuid.uuid4()),
            "clients": ["client1", "client2"],
            "min_clients": 2,
            "rounds": 1,
            "dataset_id": "test_data.csv",
            "model_arch": "SimpleMLP",
            "strategy": "FED_AVG",
            "privacy": {"mechanism": "DP_SGD", "noise_multiplier": 10.0, "max_grad_norm": 1.0, "target_epsilon": 100.0},
        }

    @pytest.fixture
    def mock_attestation_provider(self) -> Generator[MagicMock, None, None]:
        with patch("coreason_enclave.federation.executor.get_attestation_provider") as mock_get:
            provider = MagicMock()
            mock_get.return_value = provider
            # Default to TRUSTED
            provider.attest.return_value = AttestationReport(
                node_id="test_node",
                hardware_type="SIMULATION",
                enclave_signature="sig",
                measurement_hash="0" * 64,
                status="TRUSTED",
            )
            yield provider

    @pytest.fixture
    def executor(self, mock_attestation_provider: MagicMock) -> CoreasonExecutor:
        # We need to ensure the mock is active when __init__ is called
        return CoreasonExecutor(training_task_name="train_task")

    @pytest.fixture
    def mock_fl_ctx(self) -> MagicMock:
        return MagicMock(spec=FLContext)

    @pytest.fixture
    def mock_signal(self) -> MagicMock:
        signal = MagicMock(spec=Signal)
        signal.triggered = False
        return signal

    @pytest.fixture
    def valid_shareable(self, mock_job_config: Dict[str, Any]) -> Shareable:
        s = Shareable()
        s.set_header("job_config", json.dumps(mock_job_config))
        return s

    def test_secure_execution_success(
        self,
        executor: CoreasonExecutor,
        valid_shareable: Shareable,
        mock_fl_ctx: MagicMock,
        mock_signal: MagicMock,
    ) -> None:
        """Test happy path: trusted hardware, valid config, valid data."""

        # Mock DataSentry to avoid filesystem checks
        executor.sentry = MagicMock()
        executor.sentry.validate_input.return_value = True
        executor.sentry.sanitize_output.return_value = {"params": {"a": 1}}

        # Use Real DataLoader with Dummy Data
        executor.data_loader_factory = MagicMock()

        # Create dummy TensorDataset compatible with SimpleMLP (input_dim=10)
        X = torch.randn(32, 10)
        y = torch.randn(32, 1)
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=16)

        executor.data_loader_factory.get_loader.return_value = loader

        result = executor.execute("train_task", valid_shareable, mock_fl_ctx, mock_signal)

        assert result.get_return_code() == ReturnCode.OK
        # Verify flow
        executor.attestation_provider.attest.assert_called_once()  # type: ignore
        executor.data_loader_factory.get_loader.assert_called_once()
        executor.sentry.sanitize_output.assert_called_once()

    def test_secure_execution_dict_config(
        self,
        executor: CoreasonExecutor,
        mock_job_config: Dict[str, Any],
        mock_fl_ctx: MagicMock,
        mock_signal: MagicMock,
    ) -> None:
        """Test happy path with dict config (not JSON string)."""
        shareable = Shareable()
        shareable.set_header("job_config", mock_job_config)

        # Mock DataSentry
        executor.sentry = MagicMock()
        executor.sentry.validate_input.return_value = True
        executor.sentry.sanitize_output.return_value = {"params": {"a": 1}}

        # Use Real DataLoader
        executor.data_loader_factory = MagicMock()
        X = torch.randn(16, 10)
        y = torch.randn(16, 1)
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=8)

        executor.data_loader_factory.get_loader.return_value = loader

        result = executor.execute("train_task", shareable, mock_fl_ctx, mock_signal)

        assert result.get_return_code() == ReturnCode.OK

    def test_secure_execution_untrusted_hardware(
        self,
        executor: CoreasonExecutor,
        valid_shareable: Shareable,
        mock_fl_ctx: MagicMock,
        mock_signal: MagicMock,
    ) -> None:
        """Test failure when hardware is untrusted."""
        executor.attestation_provider.attest.return_value = AttestationReport(  # type: ignore
            node_id="test_node",
            hardware_type="SIMULATION",
            enclave_signature="sig",
            measurement_hash="0" * 64,
            status="UNTRUSTED",
        )

        result = executor.execute("train_task", valid_shareable, mock_fl_ctx, mock_signal)

        assert result.get_return_code() == ReturnCode.EXECUTION_EXCEPTION
        # Verify we stopped early
        # Since implementation details might vary, we check the exception was caught

    def test_secure_execution_missing_config(
        self,
        executor: CoreasonExecutor,
        mock_fl_ctx: MagicMock,
        mock_signal: MagicMock,
    ) -> None:
        """Test failure when job_config is missing."""
        executor.attestation_provider.attest.return_value.status = "TRUSTED"  # type: ignore
        shareable = Shareable()  # No header

        result = executor.execute("train_task", shareable, mock_fl_ctx, mock_signal)
        assert result.get_return_code() == ReturnCode.BAD_TASK_DATA

    def test_privacy_budget_exceeded_in_training(
        self,
        executor: CoreasonExecutor,
        valid_shareable: Shareable,
        mock_fl_ctx: MagicMock,
        mock_signal: MagicMock,
        mock_job_config: Dict[str, Any],
    ) -> None:
        """Test that execution fails if privacy budget is exceeded."""
        # Set parameters to guarantee budget fail: very small epsilon, many steps
        mock_job_config["privacy"]["target_epsilon"] = 0.0001
        mock_job_config["rounds"] = 100

        shareable = Shareable()
        shareable.set_header("job_config", json.dumps(mock_job_config))

        executor.sentry = MagicMock()
        executor.sentry.validate_input.return_value = True
        executor.sentry.sanitize_output.return_value = {"params": {}}

        # Real loader with data
        executor.data_loader_factory = MagicMock()
        X = torch.randn(32, 10)
        y = torch.randn(32, 1)
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=4)
        executor.data_loader_factory.get_loader.return_value = loader

        result = executor.execute("train_task", shareable, mock_fl_ctx, mock_signal)

        assert result.get_return_code() == ReturnCode.EXECUTION_EXCEPTION

    def test_secure_execution_invalid_config(
        self,
        executor: CoreasonExecutor,
        mock_fl_ctx: MagicMock,
        mock_signal: MagicMock,
    ) -> None:
        """Test failure when job_config is invalid."""
        executor.attestation_provider.attest.return_value.status = "TRUSTED"  # type: ignore
        shareable = Shareable()
        shareable.set_header("job_config", "INVALID_JSON")

        result = executor.execute("train_task", shareable, mock_fl_ctx, mock_signal)
        assert result.get_return_code() == ReturnCode.BAD_TASK_DATA

    def test_secure_execution_input_validation_failure(
        self,
        executor: CoreasonExecutor,
        valid_shareable: Shareable,
        mock_fl_ctx: MagicMock,
        mock_signal: MagicMock,
    ) -> None:
        """Test failure when input validation fails."""
        executor.attestation_provider.attest.return_value.status = "TRUSTED"  # type: ignore

        # Mock Sentry to raise exception
        executor.sentry = MagicMock()
        executor.sentry.validate_input.side_effect = ValueError("Invalid data")

        result = executor.execute("train_task", valid_shareable, mock_fl_ctx, mock_signal)
        assert result.get_return_code() == ReturnCode.EXECUTION_EXCEPTION

    def test_secure_execution_sanitation_failure(
        self,
        executor: CoreasonExecutor,
        valid_shareable: Shareable,
        mock_fl_ctx: MagicMock,
        mock_signal: MagicMock,
    ) -> None:
        """Test failure when output sanitation fails."""
        executor.attestation_provider.attest.return_value.status = "TRUSTED"  # type: ignore

        # Mock Sentry
        executor.sentry = MagicMock()
        executor.sentry.validate_input.return_value = True
        executor.sentry.sanitize_output.side_effect = RuntimeError("Leakage detected")

        result = executor.execute("train_task", valid_shareable, mock_fl_ctx, mock_signal)
        assert result.get_return_code() == ReturnCode.EXECUTION_EXCEPTION

    def test_secure_execution_abort_signal(
        self,
        executor: CoreasonExecutor,
        valid_shareable: Shareable,
        mock_fl_ctx: MagicMock,
        mock_signal: MagicMock,
    ) -> None:
        """Test handling of abort signal during training setup."""
        # Mock Sentry
        executor.sentry = MagicMock()
        executor.sentry.validate_input.return_value = True

        # Mock DataLoaderFactory
        executor.data_loader_factory = MagicMock()
        X = torch.randn(16, 10)
        y = torch.randn(16, 1)
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=8)

        executor.data_loader_factory.get_loader.return_value = loader

        # Trigger abort signal
        mock_signal.triggered = True

        result = executor.execute("train_task", valid_shareable, mock_fl_ctx, mock_signal)

        # If aborted, it returns an empty Shareable (OK) in current impl
        assert result.get_return_code() == ReturnCode.OK
        # Shareable might contain headers even if empty, so we check for empty data dict if applicable,
        # or just that specific keys are missing.
        assert not result.get("params")

    def test_secure_execution_privacy_init_failure(
        self,
        executor: CoreasonExecutor,
        valid_shareable: Shareable,
        mock_fl_ctx: MagicMock,
        mock_signal: MagicMock,
    ) -> None:
        """Test failure when PrivacyGuard cannot be initialized."""
        # Mock Sentry
        executor.sentry = MagicMock()
        executor.sentry.validate_input.return_value = True

        # Mock DataLoaderFactory
        executor.data_loader_factory = MagicMock()
        X = torch.randn(16, 10)
        y = torch.randn(16, 1)
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=8)

        executor.data_loader_factory.get_loader.return_value = loader

        # Mock PrivacyGuard to raise exception
        with patch("coreason_enclave.federation.executor.PrivacyGuard") as mock_guard:
            mock_guard.side_effect = ValueError("Invalid privacy config")

            result = executor.execute("train_task", valid_shareable, mock_fl_ctx, mock_signal)
            assert result.get_return_code() == ReturnCode.EXECUTION_EXCEPTION

    def test_secure_execution_consecutive_calls(
        self,
        executor: CoreasonExecutor,
        valid_shareable: Shareable,
        mock_fl_ctx: MagicMock,
        mock_signal: MagicMock,
        mock_job_config: Dict[str, Any],
    ) -> None:
        """Test consecutive calls: Valid -> Invalid -> Valid."""
        # Setup Valid
        executor.sentry = MagicMock()
        executor.sentry.validate_input.return_value = True
        executor.sentry.sanitize_output.return_value = {"params": {}}

        # Mock DataLoaderFactory
        executor.data_loader_factory = MagicMock()
        X = torch.randn(16, 10)
        y = torch.randn(16, 1)
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=8)

        executor.data_loader_factory.get_loader.return_value = loader

        # 1. Valid
        result1 = executor.execute("train_task", valid_shareable, mock_fl_ctx, mock_signal)
        assert result1.get_return_code() == ReturnCode.OK

        # 2. Invalid (Missing Config)
        bad_shareable = Shareable()
        result2 = executor.execute("train_task", bad_shareable, mock_fl_ctx, mock_signal)
        assert result2.get_return_code() == ReturnCode.BAD_TASK_DATA

        # 3. Valid (Re-use executor)
        result3 = executor.execute("train_task", valid_shareable, mock_fl_ctx, mock_signal)
        assert result3.get_return_code() == ReturnCode.OK

    def test_secure_execution_schema_boundary_values(
        self,
        executor: CoreasonExecutor,
        mock_job_config: Dict[str, Any],
        mock_fl_ctx: MagicMock,
        mock_signal: MagicMock,
    ) -> None:
        """Test config with boundary values that fail schema validation."""
        mock_job_config["rounds"] = 0  # Invalid
        shareable = Shareable()
        shareable.set_header("job_config", json.dumps(mock_job_config))

        executor.sentry = MagicMock()  # Should not be reached but good to mock

        result = executor.execute("train_task", shareable, mock_fl_ctx, mock_signal)
        assert result.get_return_code() == ReturnCode.BAD_TASK_DATA
