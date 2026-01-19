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
from typing import Dict, Optional, Tuple
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
from torch.utils.data import DataLoader

from coreason_enclave.data.loader import DataLoaderFactory
from coreason_enclave.hardware.factory import get_attestation_provider
from coreason_enclave.models.registry import ModelRegistry
from coreason_enclave.privacy import PrivacyBudgetExceededError, PrivacyGuard
from coreason_enclave.schemas import AggregationStrategy, FederationJob
from coreason_enclave.sentry import DataSentry, FileExistenceValidator
from coreason_enclave.utils.logger import logger

# Default Learning Rate for the optimizer
DEFAULT_LR = 0.01


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
            RuntimeError: If attestation fails or status is not TRUSTED.
        """
        report = self.attestation_provider.attest()
        if report.status != "TRUSTED":
            raise RuntimeError(f"Untrusted environment: {report.status}")
        logger.info(f"Environment attested: {report.hardware_type} ({report.status})")

    def _parse_job_config(self, shareable: Shareable) -> Optional[FederationJob]:
        """
        Parse and validate the FederationJob configuration from shareable.

        Args:
            shareable: The input data containing the job configuration.

        Returns:
            FederationJob: Validated configuration object, or None if parsing failed.
        """
        job_config_raw = shareable.get_header("job_config")
        if not job_config_raw:
            logger.error("Missing job_config in shareable header")
            return None

        try:
            # If it's a JSON string, parse it. If it's already a dict, use it.
            if isinstance(job_config_raw, str):
                job_config_dict = json.loads(job_config_raw)
            else:
                job_config_dict = job_config_raw

            return FederationJob(**job_config_dict)
        except Exception as e:
            logger.error(f"Invalid job_config: {e}")
            return None

    def _load_and_capture_params(
        self, model: torch.nn.Module, shareable: Shareable, job_config: FederationJob
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        Load incoming parameters into the model and optionally capture them as global parameters.

        Args:
            model: The PyTorch model to update.
            shareable: The input shareable containing 'params'.
            job_config: The job configuration.

        Returns:
            Optional[Dict[str, torch.Tensor]]: Global parameters for FedProx, or None.
        """
        incoming_params = shareable.get("params")
        global_params: Optional[Dict[str, torch.Tensor]] = None

        if incoming_params:
            try:
                # Convert list/numpy to tensor if needed
                state_dict = {
                    k: torch.tensor(v) if not isinstance(v, torch.Tensor) else v for k, v in incoming_params.items()
                }
                model.load_state_dict(state_dict)
                logger.info("Loaded incoming params into model")

                # Keep a copy of global params for FedProx
                if job_config.strategy == AggregationStrategy.FED_PROX:
                    logger.info("FedProx enabled. Capturing global params.")
                    global_params = {k: v.clone().detach() for k, v in model.named_parameters() if v.requires_grad}
            except Exception as e:
                logger.warning(f"Failed to load incoming params: {e}")
                # We log warning but proceed, assuming robust initialization or fallback logic

        return global_params

    def _calculate_proximal_loss(
        self,
        model: torch.nn.Module,
        global_params: Dict[str, torch.Tensor],
        mu: float,
    ) -> torch.Tensor:
        """
        Calculate the FedProx proximal term: (mu / 2) * ||w - w_global||^2.

        Args:
            model: The current model.
            global_params: The global parameters (snapshot).
            mu: The proximal coefficient.

        Returns:
            torch.Tensor: The scalar proximal loss term.
        """
        proximal_term = torch.tensor(0.0, device=next(model.parameters()).device)
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            # Opacus wraps model, adding _module. prefix
            clean_name = name.replace("_module.", "")
            if clean_name in global_params:
                proximal_term += (param - global_params[clean_name]).norm(2) ** 2

        return (mu / 2.0) * proximal_term

    def _run_training_loop(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        optimizer: optim.Optimizer,
        privacy_guard: PrivacyGuard,
        job_config: FederationJob,
        global_params: Optional[Dict[str, torch.Tensor]],
        abort_signal: Signal,
    ) -> Optional[Tuple[float, float]]:
        """
        Execute the main training loop.

        Args:
            model: The model.
            train_loader: Data loader.
            optimizer: Optimizer.
            privacy_guard: Privacy guard.
            job_config: Configuration.
            global_params: Global parameters for FedProx.
            abort_signal: Signal.

        Returns:
            Optional[Tuple[float, float]]: (average_loss, current_epsilon) or None if aborted/failed.
        """
        model.train()
        total_loss = 0.0
        num_batches = 0
        epochs = 1  # Local epochs per task execution

        try:
            for _epoch in range(epochs):
                for _batch_idx, (data, target) in enumerate(train_loader):
                    if abort_signal.triggered:
                        logger.info("Abort signal triggered. Stopping.")
                        return None

                    optimizer.zero_grad()
                    output = model(data)
                    loss = torch.nn.functional.mse_loss(output, target)

                    # FedProx Logic
                    if job_config.strategy == AggregationStrategy.FED_PROX and global_params is not None:
                        loss += self._calculate_proximal_loss(model, global_params, job_config.proximal_mu)

                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    num_batches += 1

                    # Check Privacy Budget
                    try:
                        privacy_guard.check_budget()
                    except PrivacyBudgetExceededError:
                        logger.critical("Privacy budget exceeded during training. Aborting.")
                        # Returning None signals failure to the caller
                        # (Caller should ideally map this to EXECUTION_EXCEPTION)
                        # But wait, we need to propagate this specific error?
                        # The original code caught it and returned EXECUTION_EXCEPTION.
                        raise

            avg_loss = total_loss / max(1, num_batches)
            epsilon = privacy_guard.get_current_epsilon()
            return avg_loss, epsilon

        except Exception as e:
            # Let caller handle exceptions
            raise e

    def _execute_training(
        self,
        shareable: Shareable,
        fl_ctx: FLContext,
        abort_signal: Signal,
    ) -> Shareable:
        """
        Orchestrate the training workflow.
        Refactored to compose smaller methods.
        """
        # 1. Check Hardware Trust
        self._check_hardware_trust()

        # 2. Parse Config
        job_config = self._parse_job_config(shareable)
        if not job_config:
            return make_reply(ReturnCode.BAD_TASK_DATA)

        # 3. Load Resources & Initialize
        try:
            dataset_id = job_config.dataset_id
            model_arch = job_config.model_arch

            # Data
            train_loader = self.data_loader_factory.get_loader(dataset_id, batch_size=32)

            # Model
            model_cls = ModelRegistry.get(model_arch)
            model = model_cls()

            # Params (FedProx capture happens here)
            global_params = self._load_and_capture_params(model, shareable, job_config)

            # Privacy & Optimizer
            optimizer = optim.SGD(model.parameters(), lr=DEFAULT_LR)
            privacy_guard = PrivacyGuard(config=job_config.privacy)
            model, optimizer, train_loader = privacy_guard.attach(model, optimizer, train_loader)

        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

        logger.info(f"Starting training execution for {job_config.rounds} rounds (local epochs)...")

        # 4. Run Training Loop
        try:
            result = self._run_training_loop(
                model, train_loader, optimizer, privacy_guard, job_config, global_params, abort_signal
            )

            if result is None:
                # Aborted manually inside loop
                return Shareable()

            avg_loss, epsilon = result

        except Exception as e:
            logger.exception(f"Training loop failed: {e}")
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

        # 5. Sanitize Output
        outgoing_params = {k: v.cpu().numpy().tolist() for k, v in model.state_dict().items()}

        training_result = {
            "params": outgoing_params,
            "metrics": {"loss": avg_loss, "epsilon": epsilon},
            "meta": {"dataset_id": dataset_id, "model": model_arch},
        }

        try:
            sanitized_result = self.sentry.sanitize_output(training_result)
        except Exception as e:  # pragma: no cover
            logger.error(f"Output sanitation failed: {e}")
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

        # 6. Return Result
        result_shareable = Shareable()
        result_shareable.update(sanitized_result)
        return result_shareable
