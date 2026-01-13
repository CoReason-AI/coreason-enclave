# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_enclave

import sys
from unittest.mock import MagicMock

# NVFlare 2.7.1 has an issue on Windows where it imports 'resource' (Unix-only).
# We mock it here to allow usage on Windows.
if sys.platform == "win32":  # pragma: no cover
    try:
        import resource  # type: ignore # noqa: F401
    except ImportError:
        sys.modules["resource"] = MagicMock()

from nvflare.apis.executor import Executor
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal

from coreason_enclave.utils.logger import logger


class CoreasonExecutor(Executor):  # type: ignore[misc]
    """
    Coreason's custom Executor for NVIDIA FLARE.
    Responsible for orchestrating the training process inside the TEE.
    """

    def __init__(
        self,
        training_task_name: str = "train",
        aggregation_task_name: str = "aggregate",
    ) -> None:
        """
        Initialize the CoreasonExecutor.

        Args:
            training_task_name: The name of the training task to listen for.
            aggregation_task_name: The name of the aggregation task.
        """
        super().__init__()
        self.training_task_name = training_task_name
        self.aggregation_task_name = aggregation_task_name
        logger.info(f"CoreasonExecutor initialized (train={training_task_name})")

    def execute(
        self,
        task_name: str,
        shareable: Shareable,
        fl_ctx: FLContext,
        abort_signal: Signal,
    ) -> Shareable:
        """
        Execute a task assigned by the server.

        Args:
            task_name: The name of the task.
            shareable: The data provided by the server (e.g., model weights).
            fl_ctx: The federation context.
            abort_signal: Signal to check for abortion.

        Returns:
            Shareable: The result of the execution (e.g., updated weights).
        """
        logger.info(f"Received task: {task_name}")

        if task_name == self.training_task_name:
            return self._execute_training(shareable, fl_ctx, abort_signal)

        logger.warning(f"Unknown task: {task_name}")
        return Shareable()

    def _execute_training(
        self,
        shareable: Shareable,
        fl_ctx: FLContext,
        abort_signal: Signal,
    ) -> Shareable:
        """
        Execute the training loop.

        Args:
            shareable: Input data.
            fl_ctx: Context.
            abort_signal: Abort signal.

        Returns:
            Shareable: Training result.
        """
        # Placeholder for training logic
        logger.info("Starting training execution...")

        # Check abort signal
        if abort_signal.triggered:
            logger.info("Abort signal triggered. Stopping.")
            return Shareable()

        return Shareable()
