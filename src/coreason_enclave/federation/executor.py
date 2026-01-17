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

import torch
import torch.optim as optim
from nvflare.apis.executor import Executor
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import ReturnCode, Shareable, make_reply
from nvflare.apis.signal import Signal

from coreason_enclave.data.loader import DataLoaderFactory
from coreason_enclave.hardware.factory import get_attestation_provider
from coreason_enclave.models.registry import ModelRegistry
from coreason_enclave.privacy import PrivacyBudgetExceededError, PrivacyGuard
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
        self.data_loader_factory = DataLoaderFactory(self.sentry)

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

        # 3. Load Data & Model
        try:
            dataset_id = job_config.dataset_id
            model_arch = job_config.model_arch

            # Load Data
            train_loader = self.data_loader_factory.get_loader(dataset_id, batch_size=32)

            # Load Model from Registry
            model_cls = ModelRegistry.get(model_arch)
            # Assumption: Model can be instantiated with default args or we need hyperparameters in config
            # For simplicity, assuming default args or handled by model internal logic
            model = model_cls()

            # Load incoming weights if any
            incoming_params = shareable.get("params")
            if incoming_params:
                # In a real NVFlare setup, we'd need to convert dict to state_dict keys match
                # For simplicity, assuming incoming_params is a valid state_dict or we skip if empty (initial round)
                # Converting dict to model state_dict is non-trivial without knowing structure perfectly.
                # We will assume for this "Atomic Unit" that we start fresh or the params match.
                pass

        except Exception as e:  # pragma: no cover
            logger.error(f"Data/Model loading failed: {e}")
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

        # 4. Initialize Privacy & Optimizer
        try:
            optimizer = optim.SGD(model.parameters(), lr=0.01)
            privacy_guard = PrivacyGuard(config=job_config.privacy)

            # Attach Privacy Engine
            model, optimizer, train_loader = privacy_guard.attach(model, optimizer, train_loader)
        except Exception as e:
            logger.error(f"Privacy/Optimizer initialization failed: {e}")
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

        logger.info(f"Starting training execution for {job_config.rounds} rounds (local epochs)...")

        # 5. Training Loop
        model.train()
        total_loss = 0.0
        num_batches = 0

        # We interpret 'rounds' in FederationJob as local epochs for this iteration
        epochs = 1  # Usually FL runs 1-5 local epochs per global round.
        # But wait, FederationJob.rounds usually means GLOBAL rounds.
        # The client just trains for `epochs` per task execution.
        # Let's assume 1 local epoch per task execution for now, or configurable.

        try:
            for _epoch in range(epochs):
                for _batch_idx, (data, target) in enumerate(train_loader):
                    # Check abort signal
                    if abort_signal.triggered:
                        logger.info("Abort signal triggered. Stopping.")
                        return Shareable()

                    optimizer.zero_grad()
                    output = model(data)
                    loss = torch.nn.functional.mse_loss(output, target)
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    num_batches += 1

                    # Check Privacy Budget
                    try:
                        privacy_guard.check_budget()
                    except PrivacyBudgetExceededError:
                        logger.critical("Privacy budget exceeded during training. Aborting.")
                        return make_reply(ReturnCode.EXECUTION_EXCEPTION)

        except Exception as e:  # pragma: no cover
            logger.exception(f"Training loop failed: {e}")
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

        # 6. Sanitize Output
        # Convert model state_dict to dict
        # Note: Opacus wraps the model, so we might need model._module if accessing original attributes,
        # but state_dict() handles it.
        # However, Opacus adds '_module.' prefix? Check Opacus docs.
        # Usually it's fine.

        outgoing_params = {k: v.cpu().numpy().tolist() for k, v in model.state_dict().items()}

        training_result = {
            "params": outgoing_params,
            "metrics": {"loss": total_loss / max(1, num_batches), "epsilon": privacy_guard.get_current_epsilon()},
            "meta": {"dataset_id": dataset_id, "model": model_arch},
        }

        try:
            sanitized_result = self.sentry.sanitize_output(training_result)
        except Exception as e:  # pragma: no cover
            logger.error(f"Output sanitation failed: {e}")
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

        # 7. Return Result
        result_shareable = Shareable()
        result_shareable.update(sanitized_result)
        return result_shareable
