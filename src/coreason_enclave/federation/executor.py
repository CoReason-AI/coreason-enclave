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
from nvflare.apis.shareable import ReturnCode, Shareable, make_reply
from nvflare.apis.signal import Signal

from coreason_enclave.hardware.factory import get_attestation_provider
from coreason_enclave.privacy import PrivacyGuard
from coreason_enclave.schemas import FederationJob
from coreason_enclave.sentry import DataSentry, FileExistenceValidator
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

        # Initialize Security Components
        self.attestation_provider = get_attestation_provider()
        # In a real scenario, we might inject a more complex validator.
        # For now, we use the FileExistenceValidator.
        self.sentry = DataSentry(validator=FileExistenceValidator())

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

        try:
            if task_name == self.training_task_name:
                return self._execute_training(shareable, fl_ctx, abort_signal)

            logger.warning(f"Unknown task: {task_name}")
            return make_reply(ReturnCode.TASK_UNKNOWN)
        except Exception as e:
            logger.exception(f"Execution failed for task {task_name}: {e}")
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

    def _check_hardware_trust(self) -> None:
        """
        Verify that the environment is trusted.

        Raises:
            SecurityError: If attestation fails or status is not TRUSTED.
        """
        report = self.attestation_provider.attest()
        if report.status != "TRUSTED":
            raise RuntimeError(f"Untrusted environment: {report.status}")
        logger.info(f"Environment attested: {report.hardware_type} ({report.status})")

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
        # 1. Check Hardware Trust
        self._check_hardware_trust()

        # 2. Parse Config
        job_config_raw = shareable.get_header("job_config")
        if not job_config_raw:
            logger.error("Missing job_config in shareable header")
            return make_reply(ReturnCode.BAD_TASK_DATA)

        try:
            # If it's a JSON string, parse it. If it's already a dict, use it.
            if isinstance(job_config_raw, str):
                job_config_dict = json.loads(job_config_raw)
            else:
                job_config_dict = job_config_raw

            job_config = FederationJob(**job_config_dict)
        except Exception as e:
            logger.error(f"Invalid job_config: {e}")
            return make_reply(ReturnCode.BAD_TASK_DATA)

        # 3. Validate Input
        # For this implementation, we assume the dataset ID is passed in the config
        # or we use a default. The schema expects client IDs, etc.
        # Let's assume there is a 'dataset_id' field in the custom props of the job
        # or we derive it. Since FederationJob doesn't have it, we might look
        # for a separate header or assume a standard path.
        # For now, let's use a dummy dataset ID to demonstrate validation.
        dataset_id = "training_data.csv"

        try:
            # We pass None as schema because FileExistenceValidator ignores it
            self.sentry.validate_input(dataset_id, schema=None)
        except Exception as e:
            logger.error(f"Input validation failed: {e}")
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

        # 4. Initialize Privacy
        try:
            privacy_guard = PrivacyGuard(config=job_config.privacy)  # noqa: F841
            # In a real loop, we would attach this to the optimizer.
        except Exception as e:
            logger.error(f"Privacy initialization failed: {e}")
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

        logger.info("Starting training execution...")

        # Check abort signal
        if abort_signal.triggered:
            logger.info("Abort signal triggered. Stopping.")
            return Shareable()

        # 5. Simulate Training (Mock)
        # Create dummy result
        training_result = {
            "params": {"layer1": [0.1, 0.2]},
            "metrics": {"loss": 0.5, "accuracy": 0.9},
            "meta": {"round": 1},
        }

        # 6. Sanitize Output
        try:
            sanitized_result = self.sentry.sanitize_output(training_result)
        except Exception as e:
            logger.error(f"Output sanitation failed: {e}")
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

        # 7. Return Result
        result_shareable = Shareable()
        result_shareable.update(sanitized_result)
        return result_shareable
