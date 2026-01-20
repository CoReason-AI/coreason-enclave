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
Federated Learning Strategies.

Implements the logic for different aggregation strategies (FedAvg, FedProx, SCAFFOLD).
These strategies hook into the training loop to apply corrections or updates.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch
from nvflare.apis.shareable import Shareable

from coreason_enclave.schemas import FederationJob
from coreason_enclave.utils.logger import logger


class TrainingStrategy(ABC):
    """Abstract base class for Federated Learning aggregation strategies."""

    @abstractmethod
    def before_training(self, model: torch.nn.Module, shareable: Shareable, job_config: FederationJob) -> None:
        """
        Called before the training loop starts.

        Use this to capture global state (e.g., global model weights) or initialize
        strategy-specific variables.

        Args:
            model: The local model (already loaded with global weights).
            shareable: The data received from the server.
            job_config: The federation job configuration.
        """
        pass  # pragma: no cover

    @abstractmethod
    def calculate_loss_correction(self, model: torch.nn.Module) -> torch.Tensor:
        """
        Calculate any additional loss term (e.g., FedProx proximal term).

        Args:
            model: The current model.

        Returns:
            torch.Tensor: The additional loss to add to the objective.
        """
        pass  # pragma: no cover

    @abstractmethod
    def after_optimizer_step(self, model: torch.nn.Module, lr: float) -> None:
        """
        Called after optimizer.step().

        Used for strategies that need to apply corrections directly to weights (e.g. SCAFFOLD).

        Args:
            model: The model.
            lr: The learning rate.
        """
        pass  # pragma: no cover

    @abstractmethod
    def after_training(self, model: torch.nn.Module, lr: float, steps: int) -> Dict[str, Any]:
        """
        Called after training completes.

        Args:
            model: The trained model.
            lr: The learning rate.
            steps: Total training steps taken.

        Returns:
            Dict[str, Any]: Additional metrics or updates to send back to the server.
        """
        pass  # pragma: no cover


class FedAvgStrategy(TrainingStrategy):
    """Standard Federated Averaging (FedAvg). No special corrections."""

    def before_training(self, model: torch.nn.Module, shareable: Shareable, job_config: FederationJob) -> None:
        """No-op for FedAvg."""
        pass

    def calculate_loss_correction(self, model: torch.nn.Module) -> torch.Tensor:
        """No loss correction for FedAvg."""
        return torch.tensor(0.0, device=next(model.parameters()).device)

    def after_optimizer_step(self, model: torch.nn.Module, lr: float) -> None:
        """No-op for FedAvg."""
        pass

    def after_training(self, model: torch.nn.Module, lr: float, steps: int) -> Dict[str, Any]:
        """No extra metrics for FedAvg."""
        return {}


class FedProxStrategy(TrainingStrategy):
    """FedProx strategy with proximal term regularization."""

    def __init__(self) -> None:
        self.global_params: Optional[Dict[str, torch.Tensor]] = None
        self.mu: float = 0.0

    def before_training(self, model: torch.nn.Module, shareable: Shareable, job_config: FederationJob) -> None:
        """Capture global parameters for the proximal term calculation."""
        self.mu = job_config.proximal_mu

        # Only capture global params if they were provided by the server.
        # If we are starting from scratch (no params), there is no "global model" to anchor to.
        if not shareable.get("params"):
            logger.warning("No params in shareable, skipping FedProx global param capture.")
            self.global_params = None
            return

        logger.info(f"FedProx enabled. Capturing global params. Mu={self.mu}")
        self.global_params = {k: v.clone().detach() for k, v in model.named_parameters() if v.requires_grad}

    def calculate_loss_correction(self, model: torch.nn.Module) -> torch.Tensor:
        """Calculate the proximal term: (mu / 2) * ||w - w_global||^2."""
        if self.global_params is None:
            return torch.tensor(0.0, device=next(model.parameters()).device)

        proximal_term = torch.tensor(0.0, device=next(model.parameters()).device)
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            clean_name = name.replace("_module.", "")
            if clean_name in self.global_params:
                proximal_term += (param - self.global_params[clean_name]).norm(2) ** 2

        return (self.mu / 2.0) * proximal_term

    def after_optimizer_step(self, model: torch.nn.Module, lr: float) -> None:
        """No-op for FedProx."""
        pass

    def after_training(self, model: torch.nn.Module, lr: float, steps: int) -> Dict[str, Any]:
        """No extra metrics for FedProx."""
        return {}


class ScaffoldStrategy(TrainingStrategy):
    """SCAFFOLD strategy with control variates to correct client drift."""

    def __init__(self, c_local: Dict[str, torch.Tensor]) -> None:
        self.c_local = c_local
        self.c_global: Dict[str, torch.Tensor] = {}
        self.global_params: Optional[Dict[str, torch.Tensor]] = None

    def before_training(self, model: torch.nn.Module, shareable: Shareable, job_config: FederationJob) -> None:
        """Initialize control variates (c_global) and capture global params."""
        logger.info("SCAFFOLD enabled. Capturing global params and controls.")

        # Only capture global params if present
        if shareable.get("params"):
            self.global_params = {k: v.clone().detach() for k, v in model.named_parameters() if v.requires_grad}
        else:
            self.global_params = None

        incoming = shareable.get("scaffold_c_global")
        if incoming:
            self.c_global = {k: torch.tensor(v) if not isinstance(v, torch.Tensor) else v for k, v in incoming.items()}

    def calculate_loss_correction(self, model: torch.nn.Module) -> torch.Tensor:
        """SCAFFOLD applies correction directly to weights, not loss."""
        return torch.tensor(0.0, device=next(model.parameters()).device)

    def after_optimizer_step(self, model: torch.nn.Module, lr: float) -> None:
        """
        Apply SCAFFOLD correction directly to parameters.
        w <- w - lr * (c_global - c_local)
        """
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            clean_name = name.replace("_module.", "")
            cg = self.c_global.get(clean_name, torch.zeros_like(param))
            cl = self.c_local.get(clean_name, torch.zeros_like(param))

            if cg.shape != param.shape or cl.shape != param.shape:
                continue

            cg = cg.to(param.device)
            cl = cl.to(param.device)

            param.data.add_(-(cg - cl) * lr)

    def after_training(self, model: torch.nn.Module, lr: float, steps: int) -> Dict[str, Any]:
        """
        Update local control variates (c_local) and return the difference (delta_c).
        c_local_new <- c_local - c_global + (1 / (K * lr)) * (w_global - w_local)
        """
        if self.global_params is None or steps <= 0:
            return {}

        c_diff = {}
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            clean_name = name.replace("_module.", "")
            if clean_name not in self.global_params:
                continue

            w_global = self.global_params[clean_name].to(param.device)
            w_local = param.detach()

            cg = self.c_global.get(clean_name, torch.zeros_like(w_local)).to(param.device)
            cl = self.c_local.get(clean_name, torch.zeros_like(w_local)).to(param.device)

            if cg.shape != w_local.shape or cl.shape != w_local.shape or w_global.shape != w_local.shape:
                continue

            # c_local_new = cl - cg + (1 / (steps * lr)) * (w_global - w_local)
            factor = 1.0 / (steps * lr)
            new_cl = cl - cg + factor * (w_global - w_local)

            self.c_local[clean_name] = new_cl.cpu()

            diff = new_cl - cl
            c_diff[clean_name] = diff.cpu().numpy().tolist()

        return {"scaffold_updates": c_diff}
