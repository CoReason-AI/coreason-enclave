# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_enclave

from typing import Tuple

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from coreason_enclave.privacy import PrivacyBudgetExceededError, PrivacyGuard
from coreason_enclave.schemas import PrivacyConfig


@pytest.fixture  # type: ignore
def valid_privacy_config() -> PrivacyConfig:
    return PrivacyConfig(
        mechanism="DP_SGD",
        noise_multiplier=1.0,
        max_grad_norm=1.0,
        target_epsilon=3.0,
    )


@pytest.fixture  # type: ignore
def simple_model_optimizer_dataloader() -> Tuple[torch.nn.Module, torch.optim.Optimizer, DataLoader]:
    model = torch.nn.Linear(10, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    data = torch.randn(100, 10)
    labels = torch.randn(100, 1)
    dataset = TensorDataset(data, labels)
    dataloader = DataLoader(dataset, batch_size=10)
    return model, optimizer, dataloader


def test_privacy_guard_initialization(valid_privacy_config: PrivacyConfig) -> None:
    guard = PrivacyGuard(valid_privacy_config)
    assert guard.config == valid_privacy_config
    assert guard._target_epsilon == 3.0
    assert guard.get_current_epsilon() == 0.0


def test_privacy_guard_attach(
    valid_privacy_config: PrivacyConfig,
    simple_model_optimizer_dataloader: Tuple[torch.nn.Module, torch.optim.Optimizer, DataLoader],
) -> None:
    guard = PrivacyGuard(valid_privacy_config)
    model, optimizer, dataloader = simple_model_optimizer_dataloader

    p_model, p_optimizer, p_dataloader = guard.attach(model, optimizer, dataloader)

    # Check if they are wrapped. Opacus wraps optimizer in DPOptimizer
    # Note: Opacus < 1.0 used DPOptimizer, newer versions might vary but it changes the class
    assert optimizer != p_optimizer
    # Check if privacy engine is tracking the optimizer
    assert guard.privacy_engine.accountant is not None


def test_privacy_budget_tracking(
    valid_privacy_config: PrivacyConfig,
    simple_model_optimizer_dataloader: Tuple[torch.nn.Module, torch.optim.Optimizer, DataLoader],
) -> None:
    # Use a standard noise multiplier.

    # Increase delta for testing or keep small
    delta = 1e-5

    guard = PrivacyGuard(valid_privacy_config)
    model, optimizer, dataloader = simple_model_optimizer_dataloader
    p_model, p_optimizer, p_dataloader = guard.attach(model, optimizer, dataloader)

    initial_epsilon = guard.get_current_epsilon(delta)
    assert initial_epsilon == 0.0

    # Simulate a training step
    batch = next(iter(p_dataloader))
    inputs, targets = batch

    p_optimizer.zero_grad()
    outputs = p_model(inputs)
    loss = torch.nn.functional.mse_loss(outputs, targets)
    loss.backward()
    p_optimizer.step()

    step_epsilon = guard.get_current_epsilon(delta)
    assert step_epsilon > initial_epsilon

    guard.check_budget(delta)  # Should pass


def test_privacy_budget_exceeded(
    simple_model_optimizer_dataloader: Tuple[torch.nn.Module, torch.optim.Optimizer, DataLoader],
) -> None:
    # Set a very low target epsilon but reasonable noise
    # We want to exceed target epsilon (e.g. 0.0001) but stay under hard limit (5.0)
    config = PrivacyConfig(
        mechanism="DP_SGD",
        noise_multiplier=1.0,
        max_grad_norm=1.0,
        target_epsilon=0.0001,
    )
    guard = PrivacyGuard(config)
    model, optimizer, dataloader = simple_model_optimizer_dataloader
    p_model, p_optimizer, p_dataloader = guard.attach(model, optimizer, dataloader)

    # Take a step
    batch = next(iter(p_dataloader))
    inputs, targets = batch
    p_optimizer.zero_grad()
    outputs = p_model(inputs)
    loss = torch.nn.functional.mse_loss(outputs, targets)
    loss.backward()
    p_optimizer.step()

    # Check budget should fail
    with pytest.raises(PrivacyBudgetExceededError) as excinfo:
        guard.check_budget(delta=1e-5)

    # It should hit the target exceeded, not hard limit
    assert "Privacy budget exceeded" in str(excinfo.value)
    assert "HARD LIMIT" not in str(excinfo.value)


def test_hard_limit_exceeded(
    simple_model_optimizer_dataloader: Tuple[torch.nn.Module, torch.optim.Optimizer, DataLoader],
) -> None:
    # Test the hard limit of 5.0
    # We set target epsilon > 5.0 to bypass the target check,
    # but the hard limit check should catch it.
    config = PrivacyConfig(
        mechanism="DP_SGD",
        noise_multiplier=0.1,
        max_grad_norm=1.0,
        target_epsilon=10.0,  # Higher than hard limit 5.0
    )
    guard = PrivacyGuard(config)
    model, optimizer, dataloader = simple_model_optimizer_dataloader
    p_model, p_optimizer, p_dataloader = guard.attach(model, optimizer, dataloader)

    inputs, targets = next(iter(p_dataloader))

    # Force epsilon to be high.
    # We can fake it by accessing internal accountant if we want deterministic test,
    # or just loop. With noise=0.1, it should grow very fast.

    p_model.train()
    # 10 steps should be enough for low noise
    for _ in range(10):
        p_optimizer.zero_grad()
        outputs = p_model(inputs)
        loss = torch.nn.functional.mse_loss(outputs, targets)
        loss.backward()
        p_optimizer.step()
        if guard.get_current_epsilon() > 5.1:
            break

    # Verify we hit the limit
    if guard.get_current_epsilon() > 5.0:
        with pytest.raises(PrivacyBudgetExceededError) as excinfo:
            guard.check_budget()
        assert "HARD LIMIT exceeded" in str(excinfo.value)
    else:
        pytest.fail(f"Could not reach epsilon > 5.0. Current: {guard.get_current_epsilon()}")


def test_get_current_epsilon_unattached(valid_privacy_config: PrivacyConfig) -> None:
    # Test line 96: if self._optimizer is None
    guard = PrivacyGuard(valid_privacy_config)
    assert guard.get_current_epsilon() == 0.0
