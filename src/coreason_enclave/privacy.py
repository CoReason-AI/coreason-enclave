# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_enclave

from typing import Any, Tuple, cast

import torch
from opacus import PrivacyEngine
from torch.utils.data import DataLoader

from coreason_enclave.schemas import PrivacyConfig
from coreason_enclave.utils.logger import logger


class PrivacyBudgetExceededError(Exception):
    """Raised when the privacy budget (epsilon) exceeds the target threshold."""

    pass


class PrivacyGuard:
    """
    Manages Differential Privacy (DP) for the training process using Opacus.
    Responsible for gradient clipping, noise injection, and budget tracking.
    """

    def __init__(self, config: PrivacyConfig) -> None:
        """
        Initialize the PrivacyGuard.

        Args:
            config: The privacy configuration (noise, clipping, epsilon).
        """
        self.config = config
        # Use RDP accountant for numerical stability
        self.privacy_engine = PrivacyEngine(accountant="rdp")
        self._optimizer: Any = None
        self._target_epsilon = config.target_epsilon
        logger.info(
            f"PrivacyGuard initialized with target_epsilon={self._target_epsilon}, "
            f"noise_multiplier={config.noise_multiplier}, "
            f"max_grad_norm={config.max_grad_norm}"
        )

    def attach(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        data_loader: DataLoader[Any],
    ) -> Tuple[torch.nn.Module, torch.optim.Optimizer, DataLoader[Any]]:
        """
        Attach the Privacy Engine to the PyTorch components.
        This wraps the optimizer to perform gradient clipping and noise addition.

        Args:
            model: The PyTorch model.
            optimizer: The PyTorch optimizer.
            data_loader: The data loader.

        Returns:
            Tuple containing the private (model, optimizer, data_loader).
        """
        logger.info("Attaching Privacy Engine to optimizer and model.")

        # Opacus make_private modifies the optimizer and model in place but also returns them.
        # We use the explicit noise_multiplier from the config.
        model, optimizer, data_loader = self.privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=data_loader,
            noise_multiplier=self.config.noise_multiplier,
            max_grad_norm=self.config.max_grad_norm,
        )

        self._optimizer = optimizer
        return model, optimizer, data_loader

    def get_current_epsilon(self, delta: float = 1e-5) -> float:
        """
        Calculate the current privacy budget spent.

        Args:
            delta: The target delta for the epsilon calculation.
                   Usually set to 1/N or smaller.

        Returns:
            The current epsilon value.
        """
        if self._optimizer is None:
            # If not attached or no steps taken, epsilon is 0
            return 0.0

        epsilon = self.privacy_engine.get_epsilon(delta)
        return cast(float, epsilon)

    def check_budget(self, delta: float = 1e-5) -> None:
        """
        Check if the privacy budget has been exceeded.

        Args:
            delta: The delta value for epsilon calculation.

        Raises:
            PrivacyBudgetExceededError: If current epsilon > target_epsilon.
        """
        current_epsilon = self.get_current_epsilon(delta)

        # Hard limit from PRD
        HARD_LIMIT = 5.0

        if current_epsilon > HARD_LIMIT:
            logger.critical(f"Privacy budget HARD LIMIT exceeded! Epsilon: {current_epsilon:.2f} > {HARD_LIMIT}")
            raise PrivacyBudgetExceededError(
                f"Privacy budget HARD LIMIT exceeded. Current: {current_epsilon:.2f}, Limit: {HARD_LIMIT}"
            )

        if current_epsilon > self._target_epsilon:
            logger.error(f"Privacy budget exceeded! Epsilon: {current_epsilon:.2f} > Target: {self._target_epsilon}")
            raise PrivacyBudgetExceededError(
                f"Privacy budget exceeded. Current: {current_epsilon:.2f}, Target: {self._target_epsilon}"
            )

        logger.info(f"Privacy budget check passed. Epsilon: {current_epsilon:.2f} / {self._target_epsilon}")
