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
from typing import Any, Dict, Generator, Iterator
from unittest.mock import MagicMock, patch

import pytest
import torch
from coreason_identity.models import UserContext
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import ReturnCode, Shareable
from nvflare.apis.signal import Signal
from torch.utils.data import DataLoader, TensorDataset

from coreason_enclave.federation.executor import CoreasonExecutor
from coreason_enclave.schemas import AttestationReport

valid_user_context = UserContext(
    user_id="test_user", username="tester", privacy_budget_spent=0.0, privacy_budget_limit=10.0
)


class TestComplexScenarios:
    @pytest.fixture
    def mock_attestation_provider(self) -> Generator[MagicMock, None, None]:
        with patch("coreason_enclave.services.get_attestation_provider") as mock_get:
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
    def executor(self, mock_attestation_provider: MagicMock) -> Generator[CoreasonExecutor, None, None]:
        # Patch DataLoaderFactory to prevent file access
        with patch("coreason_enclave.services.DataLoaderFactory") as MockFactory:
            factory = MockFactory.return_value
            # Default behavior: return dummy loader
            dataset = TensorDataset(torch.randn(10, 10), torch.randn(10, 1))
            factory.get_loader.return_value = DataLoader(dataset, batch_size=2)

            # Need to mock DataSentry too if used
            with patch("coreason_enclave.services.DataSentry") as MockSentry:
                sentry = MockSentry.return_value
                sentry.validate_input.return_value = True
                sentry.sanitize_output.return_value = {"params": {}}

                yield CoreasonExecutor(training_task_name="train_task")

    @pytest.fixture
    def mock_fl_ctx(self) -> MagicMock:
        return MagicMock(spec=FLContext)

    @pytest.fixture
    def mock_signal(self) -> MagicMock:
        signal = MagicMock(spec=Signal)
        signal.triggered = False
        return signal

    @pytest.fixture
    def basic_job_config(self) -> Dict[str, Any]:
        return {
            "job_id": str(uuid.uuid4()),
            "clients": ["client1"],
            "min_clients": 1,
            "rounds": 1,
            "dataset_id": "test_data.csv",
            "model_arch": "SimpleMLP",  # Expects input_dim=10 by default
            "strategy": "FED_AVG",
            "privacy": {"mechanism": "DP_SGD", "noise_multiplier": 1.0, "max_grad_norm": 1.0, "target_epsilon": 10.0},
        }

    def test_dimension_mismatch(
        self,
        executor: CoreasonExecutor,
        mock_fl_ctx: MagicMock,
        mock_signal: MagicMock,
        basic_job_config: Dict[str, Any],
    ) -> None:
        """
        Scenario: The dataset has 5 features, but SimpleMLP expects 10.
        Expectation: PyTorch raises RuntimeError during forward pass.
        Executor catches it and returns EXECUTION_EXCEPTION.
        """
        shareable = Shareable()
        shareable.set_header("job_config", json.dumps(basic_job_config))

        executor.sentry = MagicMock()
        executor.sentry.validate_input.return_value = True
        executor.sentry.sanitize_output.return_value = {}

        # Mock Loader with mismatched dimensions
        executor.service._async.data_loader_factory = MagicMock()
        X = torch.randn(10, 5)  # 5 features
        y = torch.randn(10, 1)
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=2)
        executor.service._async.data_loader_factory.get_loader.return_value = loader

        result = executor.execute("train_task", shareable, mock_fl_ctx, mock_signal)

        # Should fail gracefully
        assert result.get_return_code() == ReturnCode.EXECUTION_EXCEPTION

    def test_abort_during_training(
        self,
        executor: CoreasonExecutor,
        mock_fl_ctx: MagicMock,
        mock_signal: MagicMock,
        basic_job_config: Dict[str, Any],
    ) -> None:
        """
        Scenario: Abort signal is triggered in the middle of an epoch.
        Expectation: The loop terminates early and returns a Shareable (likely empty or partial).
        """
        basic_job_config["rounds"] = 5  # Run multiple epochs
        shareable = Shareable()
        shareable.set_header("job_config", json.dumps(basic_job_config))

        executor.sentry = MagicMock()
        executor.sentry.validate_input.return_value = True
        executor.sentry.sanitize_output.return_value = {"params": {}}

        # Create a custom iterator to trigger abort
        X = torch.randn(10, 10)
        y = torch.randn(10, 1)
        dataset = TensorDataset(X, y)
        real_loader = DataLoader(dataset, batch_size=2)

        class AbortingLoader:
            def __init__(self, loader: DataLoader, signal: Signal):
                self.loader = loader
                self.signal = signal
                self.dataset = loader.dataset
                self.batch_size = loader.batch_size
                self.num_workers = 0
                self.collate_fn = None
                self.pin_memory = False
                self.drop_last = False
                self.timeout = 0
                self.worker_init_fn = None
                self.multiprocessing_context = None
                self.generator = None
                self.prefetch_factor = None
                self.persistent_workers = False

            def __len__(self) -> int:
                return len(self.loader)

            def __iter__(self) -> Iterator[Any]:
                iterator = iter(self.loader)
                # First batch ok
                yield next(iterator)
                # Trigger abort
                self.signal.triggered = True
                # Yield next (loop should check signal before processing)
                yield next(iterator)

        aborting_loader = AbortingLoader(real_loader, mock_signal)

        executor.service._async.data_loader_factory = MagicMock()
        executor.service._async.data_loader_factory.get_loader.return_value = aborting_loader

        result = executor.execute("train_task", shareable, mock_fl_ctx, mock_signal)

        # If aborted, the current implementation returns `Shareable()`.
        # ReturnCode default for Shareable() is usually OK?
        # Check `make_reply` usage in `_execute_training`:
        # `if abort_signal.triggered: return Shareable()`

        assert result.get_return_code() == ReturnCode.OK
        # Should NOT contain params because it returned early empty shareable
        assert not result.get("params")

    def test_nan_data_handling(
        self,
        executor: CoreasonExecutor,
        mock_fl_ctx: MagicMock,
        mock_signal: MagicMock,
        basic_job_config: Dict[str, Any],
    ) -> None:
        """
        Scenario: Input data contains NaNs.
        Expectation: Model produces NaN loss. Executor might catch it or Opacus might fail.
        Ideally, it should fail gracefully or finish with NaN metrics, but NOT crash the agent process.
        """
        shareable = Shareable()
        shareable.set_header("job_config", json.dumps(basic_job_config))

        executor.sentry = MagicMock()
        executor.sentry.validate_input.return_value = True
        executor.sentry.sanitize_output.return_value = {"params": []}  # Mock sanitation

        executor.service._async.data_loader_factory = MagicMock()
        X = torch.randn(10, 10)
        X[0, 0] = float("nan")  # Inject NaN
        y = torch.randn(10, 1)
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=2)
        executor.service._async.data_loader_factory.get_loader.return_value = loader

        result = executor.execute("train_task", shareable, mock_fl_ctx, mock_signal)

        # It might succeed with NaN metrics, or fail if something checks for NaNs.
        # Opacus usually handles NaNs by clipping (if they are in gradients).
        # But if inputs are NaN, forward pass -> NaN output -> NaN loss -> backward -> NaN grads.
        # Clipping: norm(NaN) is NaN.
        # If executor doesn't explicitly check NaNs, it might return OK with NaNs in weights.
        # Or if it crashes, it returns EXECUTION_EXCEPTION.
        # Either is acceptable as long as the agent stays alive (caught exception).

        # In this specific codebase, we catch exceptions.
        # If no exception, RC is OK.

        if result.get_return_code() == ReturnCode.OK:
            # If OK, check that we got a result
            assert result.get("params") is not None
        else:
            assert result.get_return_code() == ReturnCode.EXECUTION_EXCEPTION
