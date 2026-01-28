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
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import ReturnCode, Shareable
from nvflare.apis.signal import Signal

from coreason_identity.models import UserContext
from coreason_enclave.federation.executor import CoreasonExecutor
from coreason_enclave.federation import executor as executor_module
from coreason_enclave.schemas import AttestationReport


class TestCoreasonExecutor:
    @pytest.fixture
    def context(self) -> UserContext:
        return UserContext(
            sub="test-user",
            email="test@coreason.ai",
            permissions=[],
            project_context="test",
        )
    @pytest.fixture
    def mock_attestation_provider(self) -> Generator[MagicMock, None, None]:
        # get_attestation_provider moved to services.py, so we mock it there if needed,
        # or we mock it where it is imported in the code under test if it was there.
        # But CoreasonExecutor uses CoreasonEnclaveService which uses get_attestation_provider.
        # Since we are mocking executor.service in tests, we might not need this anymore for execution tests,
        # but initialization might still trigger it if we don't mock Service class itself.

        # We need to mock CoreasonEnclaveServiceAsync where it is used.
        # In CoreasonExecutor.__init__, it calls CoreasonEnclaveService().
        # CoreasonEnclaveService() calls CoreasonEnclaveServiceAsync().
        # CoreasonEnclaveServiceAsync.__init__ calls get_attestation_provider().

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
        assert executor.service is not None

    def test_execute_unknown_task(
        self,
        executor: CoreasonExecutor,
        mock_fl_ctx: MagicMock,
        mock_signal: MagicMock,
        context: UserContext,
    ) -> None:
        """Test execution with an unknown task name."""
        executor_module._CURRENT_CONTEXT = context
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
        context: UserContext,
    ) -> None:
        """Test execution with the training task name."""
        executor_module._CURRENT_CONTEXT = context
        shareable = Shareable()
        shareable.set_header("job_config", json.dumps(valid_job_config))

        # Use Mock service for integration test to avoid complex async setup in this unit test
        # We test real service in test_services.py
        executor.service = MagicMock()
        executor.service.execute_training_task.return_value = {"params": {}}

        result = executor.execute("train_task", shareable, mock_fl_ctx, mock_signal)

        executor.service.execute_training_task.assert_called_once_with(
            shareable, mock_signal, context=context
        )
        assert isinstance(result, Shareable)
        assert result.get_return_code() == ReturnCode.OK

    def test_abort_signal(
        self,
        executor: CoreasonExecutor,
        mock_fl_ctx: MagicMock,
        mock_signal: MagicMock,
        valid_job_config: Dict[str, Any],
        context: UserContext,
    ) -> None:
        """Test that execution respects the abort signal."""
        executor_module._CURRENT_CONTEXT = context
        mock_signal.triggered = True
        shareable = Shareable()
        shareable.set_header("job_config", json.dumps(valid_job_config))

        # Mock Service behavior
        executor.service = MagicMock()
        # If signal triggered, service might return empty result
        executor.service.execute_training_task.return_value = {}

        result = executor.execute("train_task", shareable, mock_fl_ctx, mock_signal)
        assert isinstance(result, Shareable)
        assert result.get_return_code() == ReturnCode.OK

    def test_execute_exception_handling(
        self,
        executor: CoreasonExecutor,
        mock_fl_ctx: MagicMock,
        mock_signal: MagicMock,
        context: UserContext,
    ) -> None:
        """Test that exceptions during execution are caught and handled."""
        executor_module._CURRENT_CONTEXT = context
        shareable = Shareable()
        # Mock service to raise exception
        executor.service = MagicMock()
        executor.service.execute_training_task.side_effect = RuntimeError("Service failed")

        result = executor.execute("train_task", shareable, mock_fl_ctx, mock_signal)
        assert isinstance(result, Shareable)
        assert result.get_return_code() == ReturnCode.EXECUTION_EXCEPTION

    def test_edge_case_empty_task_name(
        self,
        executor: CoreasonExecutor,
        mock_fl_ctx: MagicMock,
        mock_signal: MagicMock,
        context: UserContext,
    ) -> None:
        """Test execution with an empty task name."""
        executor_module._CURRENT_CONTEXT = context
        shareable = Shareable()
        result = executor.execute("", shareable, mock_fl_ctx, mock_signal)
        assert result.get_return_code() == ReturnCode.TASK_UNKNOWN

    def test_configuration_edge_case(
        self,
        mock_fl_ctx: MagicMock,
        mock_signal: MagicMock,
        valid_job_config: Dict[str, Any],
        mock_attestation_provider: MagicMock,  # Need this to mock internal provider if init called here
        context: UserContext,
    ) -> None:
        """Test weird configuration: same name for training and aggregation."""
        executor_module._CURRENT_CONTEXT = context
        # Note: CoreasonExecutor calls get_attestation_provider in __init__.
        # So we need the mock active during this call.
        executor = CoreasonExecutor(training_task_name="same_name", aggregation_task_name="same_name")

        shareable = Shareable()
        shareable.set_header("job_config", json.dumps(valid_job_config))

        executor.service = MagicMock()
        executor.service.execute_training_task.return_value = {"params": {}}

        # Should behave as training since it's the first check
        result = executor.execute("same_name", shareable, mock_fl_ctx, mock_signal)
        # Since _execute_training succeeds (returns Shareable()), result RC should be OK (default)
        assert result.get_return_code() == ReturnCode.OK

    def test_close(self, executor: CoreasonExecutor) -> None:
        """Test close method releases resources."""
        executor.service = MagicMock()
        executor.close()
        executor.service.__exit__.assert_called_once()

    def test_outer_exception_handling(
        self,
        executor: CoreasonExecutor,
        mock_fl_ctx: MagicMock,
        mock_signal: MagicMock,
        context: UserContext,
    ) -> None:
        """Test that exception raised before training check is caught by outer block."""
        executor_module._CURRENT_CONTEXT = context
        # Force an exception early by making task_name comparison fail or something similar.
        # It's hard to make string comparison fail.
        # But we can mock logger to raise exception?
        # Or mock self.training_task_name to define __eq__ that raises?

        # Better: mock `self.service` to be None (and ignore type check), so attribute access raises AttributeError
        executor.service = None  # type: ignore

        # This will raise AttributeError when accessing self.service.execute_training_task inside inner block?
        # No, `if task_name == self.training_task_name:` is first.
        # If task name matches:
        # `result_dict = self.service.execute_training_task(...)` -> AttributeError.
        # This is inside inner try/except: `except Exception as e: logger... return EXECUTION_EXCEPTION`.
        # So this tests INNER exception block.

        # To test OUTER exception block, we need exception OUTSIDE inner block.
        # Inner block is:
        # try:
        #    if task == training: ...
        #    logger.warning...
        #    return TASK_UNKNOWN
        # except Exception: ...

        # Wait, the inner block covers almost everything.
        # Outer block covers `logger.warning("Received task...")`.
        # If logger raises exception, outer block catches it.

        with patch(
            "coreason_enclave.federation.executor.logger.warning", side_effect=RuntimeError("Log warn failed")
        ):
            result = executor.execute("unknown_task", Shareable(), mock_fl_ctx, mock_signal)
            assert result.get_return_code() == ReturnCode.EXECUTION_EXCEPTION
