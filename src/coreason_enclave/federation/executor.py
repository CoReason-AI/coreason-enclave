# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_enclave

"""
Coreason Executor for NVIDIA FLARE.

This module implements the custom Executor that runs inside the TEE.
It orchestrates the "Attest-Train-Aggregate" loop, managing hardware attestation,
privacy guards, and strategy execution.
"""

import sys
from typing import List, Optional
from unittest.mock import MagicMock

# NVFlare 2.7.1 has an issue on Windows where it imports 'resource' (Unix-only).
# We mock it here to allow usage on Windows.
if sys.platform == "win32":  # pragma: no cover
    try:
        import resource  # type: ignore # noqa: F401
    except ImportError:
        sys.modules["resource"] = MagicMock()

try:
    # Try importing NVFlare client logic
    import nvflare.private.fed.app.client.client_train as client_train
except ImportError:
    # Fallback for environments without nvflare installed
    client_train = None

from coreason_identity.exceptions import IdentityVerificationError
from coreason_identity.models import UserContext
from nvflare.apis.executor import Executor
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import ReturnCode, Shareable, make_reply
from nvflare.apis.signal import Signal

from coreason_enclave.privacy import PrivacyBudgetExceededError
from coreason_enclave.services import CoreasonEnclaveService
from coreason_enclave.utils.logger import logger

_CURRENT_CONTEXT: Optional[UserContext] = None


def start_client(
    context: UserContext,
    workspace: str,
    conf: str,
    opts: Optional[List[str]] = None,
) -> None:
    """
    Start the Federation Client with Identity Context.
    """
    if not context:
        raise ValueError("UserContext is required to start client")

    global _CURRENT_CONTEXT
    _CURRENT_CONTEXT = context

    logger.info(
        "Initializing Federation Executor",
        user_id=context.user_id,
        action="start_client",
    )

    # Construct NVFlare arguments
    args = [
        "coreason_enclave_wrapper",
        "-m",
        workspace,
        "-s",
        conf,
    ]
    if opts:
        args.extend(opts)

    if client_train:
        sys_argv_backup = sys.argv
        try:
            sys.argv = args
            fl_args = client_train.parse_arguments()
            client_train.main(fl_args)
        finally:
            sys.argv = sys_argv_backup
    else:
        logger.warning("NVFlare ClientTrain module not found. Skipping execution.")


def start_server(context: UserContext) -> None:
    if not context:
        raise ValueError("UserContext is required")
    logger.info(
        "Initializing Federation Executor",
        user_id=context.user_id,
        action="start_server",
    )
    # Stub implementation as per instructions to just update logic/logging
    pass


def run_federated_training(context: UserContext) -> None:
    if not context:
        raise ValueError("UserContext is required")
    logger.info(
        "Initializing Federation Executor",
        user_id=context.user_id,
        action="run_federated_training",
    )
    # Stub implementation
    pass


class CoreasonExecutor(Executor):  # type: ignore[misc]
    """
    Coreason's custom Executor for NVIDIA FLARE.

    Responsible for orchestrating the training process inside the TEE ("The Enclave Wrapper").
    It acts as the "Sightless Surgeon," training on local data without exposing it.
    """

    def __init__(
        self,
        training_task_name: str = "train",
        aggregation_task_name: str = "aggregate",
    ) -> None:
        """
        Initialize the CoreasonExecutor.

        Args:
            training_task_name (str): The name of the training task to listen for.
            aggregation_task_name (str): The name of the aggregation task.
        """
        super().__init__()
        self.training_task_name = training_task_name
        self.aggregation_task_name = aggregation_task_name

        # Initialize Service
        self.service = CoreasonEnclaveService()
        # Start lifecycle
        self.service.__enter__()

        logger.info(f"CoreasonExecutor initialized (train={training_task_name})")

    def __del__(self) -> None:  # pragma: no cover
        """Cleanup service."""
        try:
            # We can only clean up if we are not in a precarious state (e.g. interpreter shutdown).
            # With BlockingPortal, closing is sync and cleaner than anyio.run.
            self.service.__exit__(None, None, None)
        except Exception:
            pass

    def close(self) -> None:
        """Explicitly release resources."""
        self.service.__exit__(None, None, None)

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
            task_name (str): The name of the task.
            shareable (Shareable): The data provided by the server (e.g., model weights).
            fl_ctx (FLContext): The federation context.
            abort_signal (Signal): Signal to check for abortion.

        Returns:
            Shareable: The result of the execution (e.g., updated weights).
        """
        logger.info(f"Received task: {task_name}")

        # Retrieve Context
        global _CURRENT_CONTEXT
        context = _CURRENT_CONTEXT
        if not context:
            logger.error("Identity Context missing. Cannot execute task.")
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

        try:
            if task_name == self.training_task_name:
                try:
                    result_dict = self.service.execute_training_task(shareable, abort_signal, context=context)
                    # Convert dict to Shareable
                    result = Shareable()
                    result.update(result_dict)
                    return result
                except (IdentityVerificationError, PrivacyBudgetExceededError) as e:
                    logger.error(f"Security/Privacy violation: {e}")
                    return make_reply(ReturnCode.EXECUTION_RESULT_ERROR)
                except ValueError:
                    return make_reply(ReturnCode.BAD_TASK_DATA)
                except Exception as e:
                    logger.exception(f"Service execution failed: {e}")
                    return make_reply(ReturnCode.EXECUTION_EXCEPTION)

            logger.warning(f"Unknown task: {task_name}")
            return make_reply(ReturnCode.TASK_UNKNOWN)
        except Exception as e:
            logger.exception(f"Execution failed for task {task_name}: {e}")
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)
