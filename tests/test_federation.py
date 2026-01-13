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
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal

from coreason_enclave.federation.executor import CoreasonExecutor


class TestCoreasonExecutor:
    @pytest.fixture  # type: ignore[misc]
    def executor(self) -> CoreasonExecutor:
        return CoreasonExecutor(training_task_name="train_task")

    @pytest.fixture  # type: ignore[misc]
    def mock_fl_ctx(self) -> MagicMock:
        return MagicMock(spec=FLContext)

    @pytest.fixture  # type: ignore[misc]
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
        # Should return empty shareable or error, depending on logic.
        # Current logic returns empty Shareable.

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
