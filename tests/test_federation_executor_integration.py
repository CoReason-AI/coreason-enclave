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
            "user_context": {
                "user_id": "u1",
                "username": "user1",
                "privacy_budget_spent": 0.0,
                "privacy_budget_limit": 10.0,
            },
        }

    @pytest.fixture
    def executor(self) -> Generator[CoreasonExecutor, None, None]:
        # We patch dependencies of CoreasonEnclaveServiceAsync.
        # Dependencies: get_attestation_provider, DataSentry, DataLoaderFactory.
        # These are instantiated in CoreasonEnclaveServiceAsync.__init__

        # With Singleton, we must ensure these patches apply to the instance used by Executor.
        # The easiest way is to patch them BEFORE get_instance is called (which happens in executor init).
        # And rely on the autouse fixture to reset the singleton.

        with (
            patch("coreason_enclave.services.get_attestation_provider") as mock_get_attestation,
            patch("coreason_enclave.services.DataSentry") as MockDataSentry,
            patch("coreason_enclave.services.DataLoaderFactory") as MockDataLoaderFactory,
            patch("coreason_enclave.services.PrivacyGuard") as MockPrivacyGuard,
        ):
            # Setup default behavior for Attestation
            provider = MagicMock()
            mock_get_attestation.return_value = provider
            provider.attest.return_value = AttestationReport(
                node_id="test_node",
                hardware_type="SIMULATION",
                enclave_signature="sig",
                measurement_hash="0" * 64,
                status="TRUSTED",
            )

            # Setup default behavior for Sentry
            sentry_instance = MockDataSentry.return_value
            sentry_instance.validate_input.return_value = True
            sentry_instance.sanitize_output.return_value = {"params": {"a": 1}}

            # Setup default behavior for DataLoader
            loader_factory_instance = MockDataLoaderFactory.return_value
            # Default empty loader? Or we set it in test.

            executor = CoreasonExecutor(training_task_name="train_task")

            # CRITICAL: For Singleton, we must ensure we are mocking the attributes on the *active* service instance.
            # CoreasonExecutor calls `get_instance()` internally.
            # service_instance = executor.service._async

            # Since patches above mock classes, the service's attributes (assigned in __init__)
            # should already be these mocks if the service was initialized inside this with block.
            # The reset_enclave_singleton fixture ensures service is None before we start.
            # So `executor = CoreasonExecutor(...)` triggers `CoreasonEnclaveService()` which
            # triggers `CoreasonEnclaveServiceAsync()`.
            # Inside `__init__`, it calls `get_attestation_provider()` (mocked), `DataSentry()` (mocked), etc.

            # So `service_instance.attestation_provider` IS `provider`.
            # `service_instance.sentry` IS `sentry_instance`.

            # However, for tests that want to change behavior (e.g. return different loader),
            # they often modify `executor.mock_loader_factory`.
            # We must bind these helpers to the executor fixture for the tests to use.

            executor.mock_attestation_provider = provider
            executor.mock_sentry = sentry_instance
            executor.mock_loader_factory = loader_factory_instance
            executor.mock_privacy_guard = MockPrivacyGuard

            yield executor

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

        # Setup DataLoader for this specific test
        X = torch.randn(32, 10)
        y = torch.randn(32, 1)
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=16)
        executor.mock_loader_factory.get_loader.return_value = loader

        # Mock Privacy Guard behavior (attach returns model, optimizer, loader)
        # We need to ensure attach returns a tuple
        executor.mock_privacy_guard.return_value.attach.side_effect = lambda m, o, ld: (m, o, ld)
        executor.mock_privacy_guard.return_value.get_current_epsilon.return_value = 1.0

        result = executor.execute("train_task", valid_shareable, mock_fl_ctx, mock_signal)

        assert result.get_return_code() == ReturnCode.OK

        # Verify flow
        executor.mock_attestation_provider.attest.assert_called_once()
        executor.mock_loader_factory.get_loader.assert_called_once()
        executor.mock_sentry.sanitize_output.assert_called_once()

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

        # Setup DataLoader
        X = torch.randn(16, 10)
        y = torch.randn(16, 1)
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=8)
        executor.mock_loader_factory.get_loader.return_value = loader

        executor.mock_privacy_guard.return_value.attach.side_effect = lambda m, o, ld: (m, o, ld)
        executor.mock_privacy_guard.return_value.get_current_epsilon.return_value = 1.0

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
        executor.mock_attestation_provider.attest.return_value = AttestationReport(
            node_id="test_node",
            hardware_type="SIMULATION",
            enclave_signature="sig",
            measurement_hash="0" * 64,
            status="UNTRUSTED",
        )

        result = executor.execute("train_task", valid_shareable, mock_fl_ctx, mock_signal)

        assert result.get_return_code() == ReturnCode.EXECUTION_EXCEPTION

    def test_secure_execution_missing_config(
        self,
        executor: CoreasonExecutor,
        mock_fl_ctx: MagicMock,
        mock_signal: MagicMock,
    ) -> None:
        """Test failure when job_config is missing."""
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
        # Note: Since we are mocking PrivacyGuard, we need to mock it raising the exception
        # The integration test logic in original file seemed to rely on real PrivacyGuard behavior?
        # "Set parameters to guarantee budget fail: very small epsilon, many steps"
        # If we mock PrivacyGuard, we must simulate this behavior.
        # If we want to use Real PrivacyGuard, we shouldn't patch it.
        # But PrivacyGuard depends on Opacus which might be heavy.
        # Let's check if we can use Real PrivacyGuard.
        # But for this test, forcing the exception via mock is cleaner for "Integration of failure handling".

        # We'll use the mock to raise exception on check_budget
        from coreason_enclave.privacy import PrivacyBudgetExceededError

        executor.mock_privacy_guard.return_value.attach.side_effect = lambda m, o, ld: (m, o, ld)
        executor.mock_privacy_guard.return_value.check_budget.side_effect = PrivacyBudgetExceededError(
            "Budget exceeded"
        )

        # Setup DataLoader
        X = torch.randn(32, 10)
        y = torch.randn(32, 1)
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=4)
        executor.mock_loader_factory.get_loader.return_value = loader

        result = executor.execute("train_task", valid_shareable, mock_fl_ctx, mock_signal)

        assert result.get_return_code() == ReturnCode.EXECUTION_RESULT_ERROR

    def test_secure_execution_invalid_config(
        self,
        executor: CoreasonExecutor,
        mock_fl_ctx: MagicMock,
        mock_signal: MagicMock,
    ) -> None:
        """Test failure when job_config is invalid."""
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
        # Note: In executor logic, ValueErrors are caught and result in BAD_TASK_DATA
        # This differs from generic EXECUTION_EXCEPTION.

        executor.mock_loader_factory.get_loader.side_effect = ValueError("Invalid data")

        result = executor.execute("train_task", valid_shareable, mock_fl_ctx, mock_signal)
        assert result.get_return_code() == ReturnCode.BAD_TASK_DATA

    def test_secure_execution_sanitation_failure(
        self,
        executor: CoreasonExecutor,
        valid_shareable: Shareable,
        mock_fl_ctx: MagicMock,
        mock_signal: MagicMock,
    ) -> None:
        """Test failure when output sanitation fails."""
        # Setup Success for other parts
        executor.mock_privacy_guard.return_value.attach.side_effect = lambda m, o, ld: (m, o, ld)
        executor.mock_privacy_guard.return_value.get_current_epsilon.return_value = 1.0

        X = torch.randn(16, 10)
        y = torch.randn(16, 1)
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=8)
        executor.mock_loader_factory.get_loader.return_value = loader

        # Fail Sanitation
        executor.mock_sentry.sanitize_output.side_effect = RuntimeError("Leakage detected")

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
        # Setup Success
        executor.mock_privacy_guard.return_value.attach.side_effect = lambda m, o, ld: (m, o, ld)

        X = torch.randn(16, 10)
        y = torch.randn(16, 1)
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=8)
        executor.mock_loader_factory.get_loader.return_value = loader

        # Trigger abort signal
        mock_signal.triggered = True

        result = executor.execute("train_task", valid_shareable, mock_fl_ctx, mock_signal)

        # If aborted, it returns an empty Shareable (OK)
        assert result.get_return_code() == ReturnCode.OK
        # And params should be missing
        assert not result.get("params")

    def test_secure_execution_privacy_init_failure(
        self,
        executor: CoreasonExecutor,
        valid_shareable: Shareable,
        mock_fl_ctx: MagicMock,
        mock_signal: MagicMock,
    ) -> None:
        """Test failure when PrivacyGuard cannot be initialized."""

        X = torch.randn(16, 10)
        y = torch.randn(16, 1)
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=8)
        executor.mock_loader_factory.get_loader.return_value = loader

        # Mock PrivacyGuard to raise exception on attach
        # ValueErrors result in BAD_TASK_DATA
        executor.mock_privacy_guard.return_value.attach.side_effect = ValueError("Invalid privacy config")

        result = executor.execute("train_task", valid_shareable, mock_fl_ctx, mock_signal)
        assert result.get_return_code() == ReturnCode.BAD_TASK_DATA

    def test_secure_execution_consecutive_calls(
        self,
        executor: CoreasonExecutor,
        valid_shareable: Shareable,
        mock_fl_ctx: MagicMock,
        mock_signal: MagicMock,
        mock_job_config: Dict[str, Any],
    ) -> None:
        """Test consecutive calls: Valid -> Invalid -> Valid."""
        # Setup Common
        executor.mock_privacy_guard.return_value.attach.side_effect = lambda m, o, ld: (m, o, ld)
        executor.mock_privacy_guard.return_value.get_current_epsilon.return_value = 1.0

        X = torch.randn(16, 10)
        y = torch.randn(16, 1)
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=8)
        executor.mock_loader_factory.get_loader.return_value = loader

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

        result = executor.execute("train_task", shareable, mock_fl_ctx, mock_signal)
        assert result.get_return_code() == ReturnCode.BAD_TASK_DATA
