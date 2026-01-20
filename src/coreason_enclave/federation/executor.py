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
from coreason_enclave.federation.strategies import (
    FedAvgStrategy,
    FedProxStrategy,
    ScaffoldStrategy,
    TrainingStrategy,
)
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

        # Initialize Security Components
        self.attestation_provider = get_attestation_provider()
        # In a real scenario, we might inject a more complex validator.
        # For now, we use the FileExistenceValidator.
        self.sentry = DataSentry(validator=FileExistenceValidator())
        self.data_loader_factory = DataLoaderFactory(self.sentry)

        # SCAFFOLD: Local Control Variate (c_local)
        # Dictionary mapping parameter names to tensors.
        self.scaffold_c_local: Dict[str, torch.Tensor] = {}

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
            task_name (str): The name of the task.
            shareable (Shareable): The data provided by the server (e.g., model weights).
            fl_ctx (FLContext): The federation context.
            abort_signal (Signal): Signal to check for abortion.

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

        Performs Remote Attestation ("The Handshake").

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
            shareable (Shareable): The input data containing the job configuration.

        Returns:
            Optional[FederationJob]: Validated configuration object, or None if parsing failed.
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

    def _get_strategy(self, job_config: FederationJob) -> TrainingStrategy:
        """
        Factory method to get the training strategy.

        Args:
            job_config (FederationJob): The federation job configuration.

        Returns:
            TrainingStrategy: The instantiated strategy (FedAvg, FedProx, or SCAFFOLD).
        """
        if job_config.strategy == AggregationStrategy.FED_AVG:
            return FedAvgStrategy()
        elif job_config.strategy == AggregationStrategy.FED_PROX:
            return FedProxStrategy()
        elif job_config.strategy == AggregationStrategy.SCAFFOLD:
            return ScaffoldStrategy(self.scaffold_c_local)
        else:
            raise ValueError(f"Unknown strategy: {job_config.strategy}")

    def _run_training_loop(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        optimizer: optim.Optimizer,
        privacy_guard: PrivacyGuard,
        strategy: TrainingStrategy,
        abort_signal: Signal,
    ) -> Optional[Tuple[float, float, int]]:
        """
        Execute the main training loop.

        Args:
            model (torch.nn.Module): The model.
            train_loader (DataLoader): Data loader.
            optimizer (optim.Optimizer): Optimizer.
            privacy_guard (PrivacyGuard): Privacy guard.
            strategy (TrainingStrategy): The active training strategy.
            abort_signal (Signal): Signal.

        Returns:
            Optional[Tuple[float, float, int]]: (avg_loss, epsilon, total_steps) or None if aborted.
        """
        model.train()
        total_loss = 0.0
        num_batches = 0
        total_steps = 0
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

                    # Strategy-specific loss correction (e.g., FedProx)
                    loss += strategy.calculate_loss_correction(model)

                    loss.backward()
                    optimizer.step()

                    # Strategy-specific post-step logic (e.g., SCAFFOLD)
                    strategy.after_optimizer_step(model, DEFAULT_LR)

                    total_loss += loss.item()
                    num_batches += 1
                    total_steps += 1

                    # Check Privacy Budget
                    try:
                        privacy_guard.check_budget()
                    except PrivacyBudgetExceededError:
                        logger.critical("Privacy budget exceeded during training. Aborting.")
                        raise

            avg_loss = total_loss / max(1, num_batches)
            epsilon = privacy_guard.get_current_epsilon()
            return avg_loss, epsilon, total_steps

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

        Steps:
        1. Hardware Attestation.
        2. Config Parsing.
        3. Input Validation (Data Sentry).
        4. Strategy & Privacy Initialization.
        5. Training Loop.
        6. Output Sanitization (Data Sentry).
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

            # Initialize Strategy
            strategy = self._get_strategy(job_config)

            # Params
            incoming_params = shareable.get("params")
            if incoming_params:
                try:
                    state_dict = {
                        k: torch.tensor(v) if not isinstance(v, torch.Tensor) else v for k, v in incoming_params.items()
                    }
                    model.load_state_dict(state_dict)
                    logger.info("Loaded incoming params into model")
                except Exception as e:
                    logger.warning(f"Failed to load incoming params: {e}")

            # Strategy Hook: Before Training (e.g., capture global params)
            strategy.before_training(model, shareable, job_config)

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
                model,
                train_loader,
                optimizer,
                privacy_guard,
                strategy,
                abort_signal,
            )

            if result is None:
                # Aborted manually inside loop
                return Shareable()

            avg_loss, epsilon, total_steps = result

        except Exception as e:
            logger.exception(f"Training loop failed: {e}")
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

        # 5. Strategy Hook: After Training (e.g., SCAFFOLD updates)
        extra_metrics = strategy.after_training(model, DEFAULT_LR, total_steps)

        # 6. Sanitize Output
        outgoing_params = {k: v.cpu().numpy().tolist() for k, v in model.state_dict().items()}

        training_result = {
            "params": outgoing_params,
            "metrics": {"loss": avg_loss, "epsilon": epsilon},
            "meta": {"dataset_id": dataset_id, "model": model_arch},
        }
        training_result.update(extra_metrics)

        try:
            sanitized_result = self.sentry.sanitize_output(training_result)
        except Exception as e:  # pragma: no cover
            logger.error(f"Output sanitation failed: {e}")
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

        # 7. Return Result
        result_shareable = Shareable()
        result_shareable.update(sanitized_result)
        return result_shareable
