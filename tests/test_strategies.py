# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_enclave

from typing import Dict
from unittest.mock import MagicMock

import pytest
import torch
from nvflare.apis.shareable import Shareable

from coreason_enclave.federation.executor import CoreasonExecutor
from coreason_enclave.federation.strategies import FedProxStrategy, ScaffoldStrategy
from coreason_enclave.schemas import AggregationStrategy, FederationJob, PrivacyConfig


class TestStrategies:
    @pytest.fixture
    def basic_job_config(self) -> FederationJob:
        return FederationJob(
            job_id="00000000-0000-0000-0000-000000000000",
            clients=["client1"],
            min_clients=1,
            rounds=10,
            dataset_id="test_dataset",
            model_arch="SimpleMLP",
            strategy=AggregationStrategy.FED_PROX,
            proximal_mu=0.1,
            privacy=PrivacyConfig(noise_multiplier=1.0, max_grad_norm=1.0, target_epsilon=10.0),
        )

    def test_fedprox_loss_correction(self, basic_job_config: FederationJob) -> None:
        strategy = FedProxStrategy()
        model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(1.0)

        # Mock Shareable (params handled by executor, but strategy captures current state)
        shareable = Shareable()
        # Strategy requires "params" key to be present to trigger capture
        shareable["params"] = {"dummy": 1}

        # Before Training: Capture global params (currently 1.0)
        strategy.before_training(model, shareable, basic_job_config)

        # Modify model weights to simulate training drift
        with torch.no_grad():
            model.weight.fill_(1.5)

        # Calculate Loss Correction: (mu / 2) * ||w - w_global||^2
        # (0.1 / 2) * (1.5 - 1.0)^2 = 0.05 * 0.25 = 0.0125
        correction = strategy.calculate_loss_correction(model)
        assert torch.isclose(correction, torch.tensor(0.0125))

    def test_scaffold_param_update(self, basic_job_config: FederationJob) -> None:
        # Setup
        c_local = {"weight": torch.tensor([[0.2]])}
        strategy = ScaffoldStrategy(c_local)

        model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(1.0)

        # Mock Shareable with c_global
        shareable = Shareable()
        shareable["params"] = {"dummy": 1}
        shareable["scaffold_c_global"] = {"weight": torch.tensor([[0.5]])}

        # Before Training
        strategy.before_training(model, shareable, basic_job_config)

        # After Optimizer Step (Correction)
        # w = w - lr * (cg - cl)
        # w = 1.0 - 0.1 * (0.5 - 0.2) = 1.0 - 0.03 = 0.97
        lr = 0.1
        strategy.after_optimizer_step(model, lr)

        expected = torch.tensor([[0.97]])
        assert torch.allclose(model.weight, expected)

    def test_scaffold_update_controls(self, basic_job_config: FederationJob) -> None:
        # Setup
        c_local = {"weight": torch.tensor([[0.0]])}
        strategy = ScaffoldStrategy(c_local)

        model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(0.9)  # Local weight after training

        shareable = Shareable()
        shareable["params"] = {"dummy": 1}
        shareable["scaffold_c_global"] = {"weight": torch.tensor([[0.1]])}

        # Capture "Global Params" (Initial state)
        # Use before_training to setup state correctly
        # We need model to be at "global" state (1.0) during capture
        with torch.no_grad():
            model.weight.fill_(1.0)
        strategy.before_training(model, shareable, basic_job_config)

        # Move model to "local" state (0.9)
        with torch.no_grad():
            model.weight.fill_(0.9)

        # After Training
        # steps = 1, lr = 0.1
        # new_cl = cl - cg + (1 / (steps * lr)) * (w_global - w_local)
        # new_cl = 0.0 - 0.1 + (1 / 0.1) * (1.0 - 0.9) = -0.1 + 10 * 0.1 = 0.9
        metrics = strategy.after_training(model, lr=0.1, steps=1)

        assert "scaffold_updates" in metrics
        diff = metrics["scaffold_updates"]["weight"][0][0]
        # diff = new_cl - old_cl = 0.9 - 0.0 = 0.9
        assert abs(diff - 0.9) < 1e-6
        assert torch.allclose(c_local["weight"], torch.tensor([[0.9]]))

    def test_scaffold_initialization(self, basic_job_config: FederationJob) -> None:
        """Test initialization branches in before_training."""
        model = torch.nn.Linear(1, 1)
        c_local: Dict[str, torch.Tensor] = {}
        strategy = ScaffoldStrategy(c_local)

        # Case 1: No params, No scaffold_c_global
        shareable = Shareable()
        strategy.before_training(model, shareable, basic_job_config)
        assert strategy.global_params is None
        assert strategy.c_global == {}

        # Case 2: Params present, No scaffold_c_global
        shareable["params"] = {"dummy": 1}
        strategy.before_training(model, shareable, basic_job_config)
        assert strategy.global_params is not None
        assert strategy.c_global == {}

    def test_scaffold_shape_mismatch(self, basic_job_config: FederationJob) -> None:
        """Test robustness against shape mismatches."""
        c_local = {"weight": torch.zeros(1, 5)}  # Mismatch (model is 1x1)
        strategy = ScaffoldStrategy(c_local)
        model = torch.nn.Linear(1, 1, bias=False)  # 1x1

        shareable = Shareable()
        shareable["params"] = {"dummy": 1}
        shareable["scaffold_c_global"] = {"weight": torch.zeros(1, 5)}  # Mismatch

        strategy.before_training(model, shareable, basic_job_config)

        # Should proceed without error (skipping update)
        strategy.after_optimizer_step(model, lr=0.1)

        metrics = strategy.after_training(model, lr=0.1, steps=1)
        assert metrics.get("scaffold_updates") == {}

    def test_scaffold_missing_globals(self, basic_job_config: FederationJob) -> None:
        """Test behavior when global params are missing."""
        c_local: Dict[str, torch.Tensor] = {}
        strategy = ScaffoldStrategy(c_local)
        model = torch.nn.Linear(1, 1)

        # global_params is None
        metrics = strategy.after_training(model, lr=0.1, steps=1)
        assert metrics == {}

        # global_params present but key missing
        shareable = Shareable()
        shareable["params"] = {"dummy": 1}
        strategy.before_training(model, shareable, basic_job_config)
        # Manually clear global params for the key
        strategy.global_params = {}

        metrics = strategy.after_training(model, lr=0.1, steps=1)
        assert metrics.get("scaffold_updates") == {}

    def test_scaffold_frozen_params(self, basic_job_config: FederationJob) -> None:
        """Test SCAFFOLD with frozen parameters (requires_grad=False)."""
        c_local = {"weight": torch.zeros(1, 2), "bias": torch.zeros(1)}
        strategy = ScaffoldStrategy(c_local)

        model = torch.nn.Linear(2, 1)
        # Freeze bias
        model.bias.requires_grad = False

        # c_global
        shareable = Shareable()
        shareable["params"] = {"dummy": 1}
        shareable["scaffold_c_global"] = {"weight": torch.zeros(1, 2), "bias": torch.zeros(1)}

        strategy.before_training(model, shareable, basic_job_config)

        # 1. Test after_optimizer_step with frozen param
        # Should hit 'continue' for bias
        strategy.after_optimizer_step(model, lr=0.1)

        # 2. Test after_training with frozen param
        # Should hit 'continue' for bias
        metrics = strategy.after_training(model, lr=0.1, steps=1)

        # Should contain weight but NOT bias
        updates = metrics.get("scaffold_updates", {})
        assert "weight" in updates
        assert "bias" not in updates

    def test_executor_unknown_strategy(self, basic_job_config: FederationJob) -> None:
        """Test executor raises error on unknown strategy."""
        executor = CoreasonExecutor()

        # Mock dependencies
        executor.attestation_provider = MagicMock()
        executor.attestation_provider.attest.return_value.status = "TRUSTED"
        executor.attestation_provider.attest.return_value.hardware_type = "TEST"
        executor.data_loader_factory = MagicMock()

        # Create a job config with unknown strategy
        # Using a raw dict to bypass Pydantic validation for the sake of this unit test
        # (Assuming we somehow got past pydantic or modified it internally)
        # Actually, Pydantic prevents this at the boundary. But _get_strategy has an else block.
        # We can just call _get_strategy directly.

        job = basic_job_config.model_copy()
        # Hack the enum value (since it's an enum, we might need to mock or cast)
        # Ideally this is unreachable if Pydantic does its job, but for coverage of the 'else':
        job.strategy = "UNKNOWN_STRATEGY"  # type: ignore

        with pytest.raises(ValueError, match="Unknown strategy"):
            executor._get_strategy(job)
