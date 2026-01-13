# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_enclave

from unittest.mock import MagicMock

import pytest
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import ReturnCode, Shareable
from nvflare.apis.signal import Signal

from coreason_enclave.federation.executor import CoreasonExecutor


class TestCoreasonExecutor:
    @pytest.fixture
    def executor(self) -> CoreasonExecutor:
        return CoreasonExecutor(training_task_name="train_task")

    @pytest.fixture
    def mock_fl_ctx(self) -> MagicMock:
        return MagicMock(spec=FLContext)

    @pytest.fixture
    def mock_signal(self) -> MagicMock:
        signal = MagicMock(spec=Signal)
        signal.triggered = False
        return signal

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
    ) -> None:
        """Test execution with the training task name."""
        shareable = Shareable()
        result = executor.execute("train_task", shareable, mock_fl_ctx, mock_signal)
        assert isinstance(result, Shareable)
        # Default implementation just returns Shareable(), implying OK if not set?
        # nvflare default shareable is OK.

    def test_abort_signal(
        self,
        executor: CoreasonExecutor,
        mock_fl_ctx: MagicMock,
        mock_signal: MagicMock,
    ) -> None:
        """Test that execution respects the abort signal."""
        mock_signal.triggered = True
        shareable = Shareable()
        result = executor.execute("train_task", shareable, mock_fl_ctx, mock_signal)
        assert isinstance(result, Shareable)

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
    ) -> None:
        """Test weird configuration: same name for training and aggregation."""
        executor = CoreasonExecutor(training_task_name="same_name", aggregation_task_name="same_name")
        shareable = Shareable()
        # Should behave as training since it's the first check
        result = executor.execute("same_name", shareable, mock_fl_ctx, mock_signal)
        # Since _execute_training succeeds (returns Shareable()), result RC should be OK (default)
        assert result.get_return_code() == ReturnCode.OK
