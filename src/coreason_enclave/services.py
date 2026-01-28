"""
Coreason Enclave Services.

This module provides the Async-Native and Sync-Facade service classes for the Coreason Enclave.
"""

import json
from typing import Any, Dict, Optional, Tuple

import anyio
import httpx
import torch
import torch.optim as optim
from coreason_identity.models import UserContext
from nvflare.apis.shareable import Shareable
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


class CoreasonEnclaveServiceAsync:
    """
    Async-Native Coreason Enclave Service.

    Handles the core logic of the enclave, including attestation, data loading,
    and training execution, in an async-first manner.
    """

    def __init__(self, client: Optional[httpx.AsyncClient] = None):
        """
        Initialize the Async Service.

        Args:
            client (Optional[httpx.AsyncClient]): External HTTP client for connection pooling.
        """
        self._internal_client = client is None
        self._client = client or httpx.AsyncClient()

        # Initialize Security Components
        self.attestation_provider = get_attestation_provider()
        self.sentry = DataSentry(validator=FileExistenceValidator())
        self.data_loader_factory = DataLoaderFactory(self.sentry)

        # SCAFFOLD: Local Control Variate (c_local)
        # Dictionary mapping parameter names to tensors.
        self.scaffold_c_local: Dict[str, torch.Tensor] = {}

    async def __aenter__(self) -> "CoreasonEnclaveServiceAsync":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._internal_client:
            await self._client.aclose()

    async def check_hardware_trust(self) -> None:
        """
        Verify that the environment is trusted.

        Performs Remote Attestation ("The Handshake").
        Wrapped in to_thread since attestation might be blocking/IO-heavy but synchronous.
        """
        # Attestation can be IO bound or CPU bound depending on implementation.
        # Assuming IO/Blocking calls, run in thread.
        await anyio.to_thread.run_sync(self._check_hardware_trust_sync)

    def _check_hardware_trust_sync(self) -> None:
        report = self.attestation_provider.attest()
        if report.status != "TRUSTED":
            raise RuntimeError(f"Untrusted environment: {report.status}")
        logger.info(f"Environment attested: {report.hardware_type} ({report.status})")

    def _parse_job_config(self, shareable: Shareable) -> Optional[FederationJob]:
        """
        Parse and validate the FederationJob configuration from shareable.
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
        if job_config.strategy == AggregationStrategy.FED_AVG:
            return FedAvgStrategy()
        elif job_config.strategy == AggregationStrategy.FED_PROX:
            return FedProxStrategy()
        elif job_config.strategy == AggregationStrategy.SCAFFOLD:
            return ScaffoldStrategy(self.scaffold_c_local)
        else:
            raise ValueError(f"Unknown strategy: {job_config.strategy}")

    def _run_training_loop_sync(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        optimizer: optim.Optimizer,
        privacy_guard: PrivacyGuard,
        strategy: TrainingStrategy,
        abort_signal: Signal,
        epochs: int = 1,
    ) -> Optional[Tuple[float, float, int]]:
        """
        Synchronous training loop to be run in a separate thread.
        """
        model.train()
        total_loss = 0.0
        num_batches = 0
        total_steps = 0

        try:
            for _epoch in range(epochs):
                for _batch_idx, (data, target) in enumerate(train_loader):
                    if abort_signal.triggered:
                        logger.info("Abort signal triggered. Stopping.")
                        return None

                    optimizer.zero_grad()
                    output = model(data)
                    loss = torch.nn.functional.mse_loss(output, target)

                    loss += strategy.calculate_loss_correction(model)

                    loss.backward()
                    optimizer.step()

                    strategy.after_optimizer_step(model, DEFAULT_LR)

                    total_loss += loss.item()
                    num_batches += 1
                    total_steps += 1

                    try:
                        privacy_guard.check_budget()
                    except PrivacyBudgetExceededError:
                        logger.critical("Privacy budget exceeded during training. Aborting.")
                        raise

            avg_loss = total_loss / max(1, num_batches)
            epsilon = privacy_guard.get_current_epsilon()
            return avg_loss, epsilon, total_steps

        except Exception as e:
            raise e

    async def train_model(
        self,
        context: UserContext,
        job_config: FederationJob,
        params: Optional[Dict[str, Any]],
        shareable: Optional[Shareable],
        abort_signal: Signal,
    ) -> Dict[str, Any]:
        """
        Execute training model logic with identity context.
        """
        if not context:
            raise ValueError("UserContext is required for training")

        logger.info(
            "Enclave Service Request",
            user_id=context.user_id,
            operation="train_model",
            model=job_config.model_arch,
        )

        # 1. Check Hardware Trust
        try:
            await self.check_hardware_trust()
        except RuntimeError as e:
            logger.error(f"Attestation failed: {e}")
            raise e

        # 3. Load Resources & Initialize
        try:
            dataset_id = job_config.dataset_id
            model_arch = job_config.model_arch
            user_context = job_config.user_context

            # Data Loading (Sync IO, might be heavy)
            # running in thread to avoid blocking loop
            train_loader = await anyio.to_thread.run_sync(
                self.data_loader_factory.get_loader, dataset_id, user_context, 32
            )

            # Model Instantiation
            model_cls = ModelRegistry.get(model_arch)
            model = model_cls()

            # Initialize Strategy
            strategy = self._get_strategy(job_config)

            # Params
            incoming_params = params
            if incoming_params:
                # torch tensor creation is fast enough, but load_state_dict might be heavy for huge models
                state_dict = {
                    k: torch.tensor(v) if not isinstance(v, torch.Tensor) else v for k, v in incoming_params.items()
                }
                model.load_state_dict(state_dict)
                logger.info("Loaded incoming params into model")

            # Strategy Hook
            if shareable:
                strategy.before_training(model, shareable, job_config)

            # Privacy & Optimizer
            optimizer = optim.SGD(model.parameters(), lr=DEFAULT_LR)
            privacy_guard = PrivacyGuard(config=job_config.privacy, user_context=user_context)

            # Opacus attach might be CPU intensive
            model, optimizer, train_loader = await anyio.to_thread.run_sync(
                privacy_guard.attach, model, optimizer, train_loader
            )

        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            raise e

        logger.info(f"Starting training execution for {job_config.rounds} rounds (local epochs)...")

        # 4. Run Training Loop (CPU Bound)
        try:
            result = await anyio.to_thread.run_sync(
                self._run_training_loop_sync,
                model,
                train_loader,
                optimizer,
                privacy_guard,
                strategy,
                abort_signal,
                1,  # epochs
            )

            if result is None:
                return {}  # Aborted

            avg_loss, epsilon, total_steps = result

        except Exception as e:
            logger.exception(f"Training loop failed: {e}")
            raise e

        # 5. Strategy Hook
        extra_metrics = strategy.after_training(model, DEFAULT_LR, total_steps)

        # 6. Sanitize Output
        # Moving to CPU and tolist is CPU bound for large models
        def prepare_output() -> Dict[str, Any]:
            outgoing_params = {k: v.cpu().numpy().tolist() for k, v in model.state_dict().items()}
            training_result = {
                "params": outgoing_params,
                "metrics": {"loss": avg_loss, "epsilon": epsilon},
                "meta": {"dataset_id": dataset_id, "model": model_arch},
            }
            training_result.update(extra_metrics)
            return training_result

        training_result = await anyio.to_thread.run_sync(prepare_output)

        try:
            # Sanitization might involve File IO (validation), run in thread
            sanitized_result = await anyio.to_thread.run_sync(self.sentry.sanitize_output, training_result)
        except Exception as e:
            logger.error(f"Output sanitation failed: {e}")
            raise e

        return dict(sanitized_result)

    async def evaluate_model(
        self,
        context: UserContext,
        job_config: FederationJob,
        params: Optional[Dict[str, Any]],
        abort_signal: Signal,
    ) -> Dict[str, Any]:
        """
        Execute evaluation model logic with identity context.
        """
        if not context:
            raise ValueError("UserContext is required for evaluation")

        logger.info(
            "Enclave Service Request",
            user_id=context.user_id,
            operation="evaluate_model",
            model=job_config.model_arch,
        )
        # TODO: Implement evaluation logic
        logger.warning("Evaluation logic not implemented")
        return {}

    async def execute_training_task(
        self,
        shareable: Shareable,
        abort_signal: Signal,
        context: UserContext,
    ) -> Dict[str, Any]:
        """
        Orchestrate the training workflow asynchronously.

        Returns a dictionary representing the result shareable content (or error).
        """
        # 1. Parse Config
        job_config = self._parse_job_config(shareable)
        if not job_config:
            raise ValueError("Invalid job configuration")

        # 2. Delegate to train_model (which handles attestation)
        return await self.train_model(context, job_config, shareable.get("params"), shareable, abort_signal)


class CoreasonEnclaveService:
    """
    Sync Facade for Coreason Enclave Service.

    Wraps CoreasonEnclaveServiceAsync to provide a synchronous interface
    compatible with existing synchronous callers (like NVFlare).
    """

    def __init__(self, client: Optional[httpx.AsyncClient] = None):
        self._async = CoreasonEnclaveServiceAsync(client)
        self._portal: Optional[anyio.from_thread.BlockingPortal] = None
        self._portal_cm: Any = None

    def __enter__(self) -> "CoreasonEnclaveService":
        # Start a persistent event loop (portal) for the context
        self._portal_cm = anyio.from_thread.start_blocking_portal()
        self._portal = self._portal_cm.__enter__()
        self._portal.call(self._async.__aenter__)
        return self

    def __exit__(self, *args: Any) -> None:
        if self._portal:
            try:
                self._portal.call(self._async.__aexit__, *args)
            finally:
                if self._portal_cm:
                    self._portal_cm.__exit__(None, None, None)
                self._portal = None
                self._portal_cm = None

    def train_model(
        self,
        context: UserContext,
        job_config: FederationJob,
        params: Optional[Dict[str, Any]],
        shareable: Optional[Shareable],
        abort_signal: Signal,
    ) -> Dict[str, Any]:
        if not self._portal:
            raise RuntimeError("Service used outside of context manager")
        return self._portal.call(self._async.train_model, context, job_config, params, shareable, abort_signal)  # type: ignore[no-any-return]

    def evaluate_model(
        self,
        context: UserContext,
        job_config: FederationJob,
        params: Optional[Dict[str, Any]],
        abort_signal: Signal,
    ) -> Dict[str, Any]:
        if not self._portal:
            raise RuntimeError("Service used outside of context manager")
        return self._portal.call(self._async.evaluate_model, context, job_config, params, abort_signal)  # type: ignore[no-any-return]

    def execute_training_task(
        self,
        shareable: Shareable,
        abort_signal: Signal,
        context: UserContext,
    ) -> Dict[str, Any]:
        """
        Execute training task synchronously.
        """
        if not self._portal:
            raise RuntimeError("Service used outside of context manager")
        return self._portal.call(self._async.execute_training_task, shareable, abort_signal, context)  # type: ignore[no-any-return]
