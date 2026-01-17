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


class TestCoreasonExecutor:
    @pytest.fixture
    def mock_attestation_provider(self) -> Generator[MagicMock, None, None]:
        with patch("coreason_enclave.federation.executor.get_attestation_provider") as mock_get:
            provider = MagicMock()
            mock_get.return_value = provider
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
    def valid_job_config(self) -> Dict[str, Any]:
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

    def test_init(self, executor: CoreasonExecutor) -> None:
        """Test initialization of the executor."""
        assert executor.training_task_name == "train_task"

    def test_execute_unknown_task(
        self,
        executor: CoreasonExecutor,
        mock_fl_ctx: MagicMock,
        mock_signal: MagicMock,
    ) -> None:
        """Test execution with an unknown task name."""
        shareable = Shareable()
        result = executor.execute("unknown_task", shareable, mock_fl_ctx, mock_signal)
        assert isinstance(result, Shareable)
        assert result.get_return_code() == ReturnCode.TASK_UNKNOWN

    def test_execute_training_task(
        self,
        executor: CoreasonExecutor,
        mock_fl_ctx: MagicMock,
        mock_signal: MagicMock,
        valid_job_config: Dict[str, Any],
    ) -> None:
        """Test execution with the training task name."""
        shareable = Shareable()
        shareable.set_header("job_config", json.dumps(valid_job_config))

        # Mock Sentry
        executor.sentry = MagicMock()
        executor.sentry.validate_input.return_value = True
        executor.sentry.sanitize_output.return_value = {"params": {}}

        # Use Real DataLoader
        executor.data_loader_factory = MagicMock()
        X = torch.randn(2, 10)
        y = torch.randn(2, 1)
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=2)
        executor.data_loader_factory.get_loader.return_value = loader

        result = executor.execute("train_task", shareable, mock_fl_ctx, mock_signal)
        assert isinstance(result, Shareable)
        assert result.get_return_code() == ReturnCode.OK

    def test_abort_signal(
        self,
        executor: CoreasonExecutor,
        mock_fl_ctx: MagicMock,
        mock_signal: MagicMock,
        valid_job_config: Dict[str, Any],
    ) -> None:
        """Test that execution respects the abort signal."""
        mock_signal.triggered = True
        shareable = Shareable()
        shareable.set_header("job_config", json.dumps(valid_job_config))

        # Mock Sentry
        executor.sentry = MagicMock()
        executor.sentry.validate_input.return_value = True

        # Use Real DataLoader
        executor.data_loader_factory = MagicMock()
        X = torch.randn(2, 10)
        y = torch.randn(2, 1)
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=2)
        executor.data_loader_factory.get_loader.return_value = loader

        result = executor.execute("train_task", shareable, mock_fl_ctx, mock_signal)
        assert isinstance(result, Shareable)
        assert result.get_return_code() == ReturnCode.OK

    def test_execute_exception_handling(
        self,
        executor: CoreasonExecutor,
        mock_fl_ctx: MagicMock,
        mock_signal: MagicMock,
    ) -> None:
        """Test that exceptions during execution are caught and handled."""
        shareable = Shareable()
        # Mock _execute_training to raise an exception
        executor._execute_training = MagicMock(side_effect=RuntimeError("Training failed"))  # type: ignore

        result = executor.execute("train_task", shareable, mock_fl_ctx, mock_signal)
        assert isinstance(result, Shareable)
        assert result.get_return_code() == ReturnCode.EXECUTION_EXCEPTION

    def test_edge_case_empty_task_name(
        self,
        executor: CoreasonExecutor,
        mock_fl_ctx: MagicMock,
        mock_signal: MagicMock,
    ) -> None:
        """Test execution with an empty task name."""
        shareable = Shareable()
        result = executor.execute("", shareable, mock_fl_ctx, mock_signal)
        assert result.get_return_code() == ReturnCode.TASK_UNKNOWN

    def test_configuration_edge_case(
        self,
        mock_fl_ctx: MagicMock,
        mock_signal: MagicMock,
        valid_job_config: Dict[str, Any],
        mock_attestation_provider: MagicMock,  # Need this to mock internal provider if init called here
    ) -> None:
        """Test weird configuration: same name for training and aggregation."""
        # Note: CoreasonExecutor calls get_attestation_provider in __init__.
        # So we need the mock active during this call.
        executor = CoreasonExecutor(training_task_name="same_name", aggregation_task_name="same_name")

        shareable = Shareable()
        shareable.set_header("job_config", json.dumps(valid_job_config))

        # Mock Sentry
        executor.sentry = MagicMock()
        executor.sentry.validate_input.return_value = True
        executor.sentry.sanitize_output.return_value = {"params": {}}

        # Use Real DataLoader
        executor.data_loader_factory = MagicMock()
        X = torch.randn(2, 10)
        y = torch.randn(2, 1)
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=2)
        executor.data_loader_factory.get_loader.return_value = loader

        # Should behave as training since it's the first check
        result = executor.execute("same_name", shareable, mock_fl_ctx, mock_signal)
        # Since _execute_training succeeds (returns Shareable()), result RC should be OK (default)
        assert result.get_return_code() == ReturnCode.OK
