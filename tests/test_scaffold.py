from typing import Any, Dict
from unittest.mock import MagicMock

import pytest
import torch
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import ReturnCode, Shareable
from nvflare.apis.signal import Signal

from coreason_enclave.federation.executor import CoreasonExecutor
from coreason_enclave.schemas import AggregationStrategy, FederationJob, PrivacyConfig


class TestScaffoldStrategy:
    @pytest.fixture
    def executor(self) -> CoreasonExecutor:
        """Fixture for CoreasonExecutor."""
        executor = CoreasonExecutor()
        # Mock dependencies to avoid side effects
        executor.attestation_provider = MagicMock()
        executor.attestation_provider.attest.return_value.status = "TRUSTED"
        executor.attestation_provider.attest.return_value.hardware_type = "TEST_HARDWARE"

        executor.data_loader_factory = MagicMock()
        # Create a dummy loader
        # SimpleMLP expects input_dim=10
        dataset = torch.utils.data.TensorDataset(torch.randn(10, 10), torch.randn(10, 1))
        loader = torch.utils.data.DataLoader(dataset, batch_size=2)
        executor.data_loader_factory.get_loader.return_value = loader

        return executor

    @pytest.fixture
    def job_config(self) -> FederationJob:
        return FederationJob(
            job_id="00000000-0000-0000-0000-000000000000",
            clients=["client1"],
            min_clients=1,
            rounds=10,
            dataset_id="test_dataset",
            model_arch="SimpleMLP",
            strategy=AggregationStrategy.SCAFFOLD,
            privacy=PrivacyConfig(noise_multiplier=1.0, max_grad_norm=1.0, target_epsilon=10.0),
        )

    def test_scaffold_correction_calculation(self, executor: CoreasonExecutor) -> None:
        """Test the mathematical correctness of the scaffold correction term applied to params."""
        model = torch.nn.Linear(2, 1, bias=False)
        # Weights = [1.0, 1.0]
        with torch.no_grad():
            model.weight.fill_(1.0)

        # c_global = [0.5, 0.5]
        c_global = {"weight": torch.tensor([[0.5, 0.5]])}

        # c_local = [0.2, 0.2]
        c_local = {"weight": torch.tensor([[0.2, 0.2]])}

        # Update Rule: w = w - lr * (cg - cl)
        # lr = 0.1
        # cg - cl = [0.3, 0.3]
        # w_new = 1.0 - 0.1 * 0.3 = 1.0 - 0.03 = 0.97

        lr = 0.1
        executor._apply_scaffold_correction(model, c_global, c_local, lr)

        expected_weight = torch.tensor([[0.97, 0.97]])
        assert torch.allclose(model.weight, expected_weight)

    def test_scaffold_update_logic(self, executor: CoreasonExecutor) -> None:
        """Test the update logic for local controls."""
        model = torch.nn.Linear(1, 1, bias=False)
        # Initial Global Weight w = 1.0
        # Final Local Weight w+ = 0.9 (simulated training update)
        # c_local = 0.0
        # c_global = 0.1
        # lr = 0.1
        # steps = 1

        lr = 0.1
        steps = 1

        # Setup initial state
        with torch.no_grad():
            model.weight.fill_(0.9)

        global_params = {"weight": torch.tensor([[1.0]])}
        c_global = {"weight": torch.tensor([[0.1]])}
        executor.scaffold_c_local = {"weight": torch.tensor([[0.0]])}

        # Formula:
        # new_cl = cl - cg + (1 / (steps * lr)) * (w - w_local)
        # new_cl = 0.0 - 0.1 + (1 / 0.1) * (1.0 - 0.9)
        # new_cl = -0.1 + 10 * 0.1
        # new_cl = -0.1 + 1.0 = 0.9

        updates = executor._update_scaffold_controls(model, c_global, global_params, lr, steps)

        # Check internal state update
        new_cl_tensor = executor.scaffold_c_local["weight"]
        assert torch.isclose(new_cl_tensor, torch.tensor([[0.9]]))

        # Check returned diff
        # diff = new_cl - old_cl = 0.9 - 0.0 = 0.9
        diff_val = updates["weight"][0][0]  # extract from list[list[float]]
        assert abs(diff_val - 0.9) < 1e-6

    def test_scaffold_edge_cases(self, executor: CoreasonExecutor) -> None:
        """Test edge cases for SCAFFOLD logic."""
        model = torch.nn.Linear(1, 1, bias=False)

        # Case 1: get_scaffold_global with None/Empty
        shareable = Shareable()
        assert executor._get_scaffold_global(shareable) == {}

        # Case 2: update_scaffold_controls with None global_params
        res = executor._update_scaffold_controls(model, {}, None, 0.1, 10)
        assert res == {}

        # Case 3: update_scaffold_controls with 0 steps
        res = executor._update_scaffold_controls(model, {}, {"weight": torch.tensor([1.0])}, 0.1, 0)
        assert res == {}

        # Case 4: Mismatched keys in global params
        global_params = {"other_layer": torch.tensor([1.0])}
        res = executor._update_scaffold_controls(model, {}, global_params, 0.1, 10)
        assert res == {}

    def test_scaffold_frozen_params(self, executor: CoreasonExecutor) -> None:
        """Test SCAFFOLD with frozen parameters (requires_grad=False)."""
        model = torch.nn.Linear(2, 1)
        # Freeze bias
        model.bias.requires_grad = False

        c_global = {"weight": torch.zeros(1, 2), "bias": torch.zeros(1)}
        c_local = {"weight": torch.zeros(1, 2), "bias": torch.zeros(1)}

        # Test _apply_scaffold_correction with frozen param
        # Should execute 'continue' for bias (no change in bias)
        executor._apply_scaffold_correction(model, c_global, c_local, 0.1)

        # Test _update_scaffold_controls with frozen param
        # Should execute 'continue' for bias
        global_params = {"weight": torch.zeros(1, 2), "bias": torch.zeros(1)}
        res = executor._update_scaffold_controls(model, c_global, global_params, 0.1, 10)

        # Should contain weight but NOT bias
        assert "weight" in res
        assert "bias" not in res

    def test_scaffold_shape_mismatch_resilience(self, executor: CoreasonExecutor) -> None:
        """Test that SCAFFOLD handles shape mismatches gracefully (avoids crash)."""
        model = torch.nn.Linear(2, 1)  # weight: [1, 2], bias: [1]

        # Inject malformed local state: wrong shape for weight [1, 5] instead of [1, 2]
        executor.scaffold_c_local = {
            "weight": torch.randn(1, 5),
            "bias": torch.randn(1)
        }

        c_global = {
            "weight": torch.randn(1, 2),
            "bias": torch.randn(1)
        }

        # Test correction calculation: Should skip 'weight' but process 'bias'
        # If it crashes, test fails.
        executor._apply_scaffold_correction(model, c_global, executor.scaffold_c_local, 0.1)

        # Test update controls: Should skip 'weight' update
        global_params = {
            "weight": torch.randn(1, 2),
            "bias": torch.randn(1)
        }
        updates = executor._update_scaffold_controls(model, c_global, global_params, 0.1, 10)

        assert "weight" not in updates
        assert "bias" in updates

    def test_scaffold_multi_round_persistence(self, executor: CoreasonExecutor, job_config: FederationJob) -> None:
        """Test complex scenario: Multiple rounds to verify state persistence."""

        # Prepare Shareable Round 1
        shareable1 = Shareable()
        shareable1.set_header("job_config", job_config.model_dump_json())

        params1 = {
            "net.0.weight": torch.randn(16, 10),
            "net.0.bias": torch.randn(16),
            "net.2.weight": torch.randn(1, 16),
            "net.2.bias": torch.randn(1),
        }
        shareable1["params"] = {k: v.numpy() for k, v in params1.items()}

        # Initial c_global is 0
        scaffold_c_global_1 = {k: torch.zeros_like(v).numpy() for k, v in params1.items()}
        shareable1["scaffold_c_global"] = scaffold_c_global_1

        # Execute Round 1
        fl_ctx = FLContext()
        abort_signal = Signal()
        result1 = executor.execute("train", shareable1, fl_ctx, abort_signal)

        assert result1.get_return_code() == ReturnCode.OK

        # Verify c_local was updated from 0 to something non-zero (due to drift)
        local_weight_r1 = executor.scaffold_c_local["net.0.weight"].clone()
        assert not torch.allclose(local_weight_r1, torch.tensor(0.0))

        # Prepare Shareable Round 2
        # Use same executor instance!
        shareable2 = Shareable()
        shareable2.set_header("job_config", job_config.model_dump_json())

        # New global params (simulated aggregation)
        params2 = {k: v + 0.1 for k, v in params1.items()}
        shareable2["params"] = {k: v.numpy() for k, v in params2.items()}

        # New c_global (simulated aggregation)
        scaffold_c_global_2 = {k: v + 0.01 for k, v in scaffold_c_global_1.items()}
        shareable2["scaffold_c_global"] = scaffold_c_global_2

        # Execute Round 2
        result2 = executor.execute("train", shareable2, fl_ctx, abort_signal)
        assert result2.get_return_code() == ReturnCode.OK

        # Verify c_local has evolved further
        local_weight_r2 = executor.scaffold_c_local["net.0.weight"]
        assert not torch.allclose(local_weight_r2, local_weight_r1)

    def test_integration_execution(self, executor: CoreasonExecutor, job_config: FederationJob) -> None:
        """Test full execution flow with SCAFFOLD strategy."""

        # Prepare Shareable
        shareable = Shareable()
        shareable.set_header("job_config", job_config.model_dump_json())

        # Mock Model Registry
        # We rely on 'SimpleMLP' being available in the registry (it's in the repo)

        # Add Params
        # SimpleMLP default: input_dim=10, hidden_dim=16, output_dim=1
        # net.0 is Linear(10, 16)
        # net.2 is Linear(16, 1)
        params: Dict[str, Any] = {
            "net.0.weight": torch.randn(16, 10),
            "net.0.bias": torch.randn(16),
            "net.2.weight": torch.randn(1, 16),
            "net.2.bias": torch.randn(1),
        }
        shareable["params"] = {k: v.numpy() for k, v in params.items()}

        # Add SCAFFOLD globals
        scaffold_c_global: Dict[str, Any] = {
            "net.0.weight": torch.zeros(16, 10),
            "net.0.bias": torch.zeros(16),
            "net.2.weight": torch.zeros(1, 16),
            "net.2.bias": torch.zeros(1),
        }
        shareable["scaffold_c_global"] = {k: v.numpy() for k, v in scaffold_c_global.items()}

        # FL Context and Signal
        fl_ctx = FLContext()
        abort_signal = Signal()

        # Execute
        result = executor.execute("train", shareable, fl_ctx, abort_signal)

        # Check Result
        assert result.get_return_code() == ReturnCode.OK
        assert "params" in result
        assert "metrics" in result
        assert "scaffold_updates" in result

        # Verify scaffold updates structure
        updates = result["scaffold_updates"]
        assert "net.0.weight" in updates
        assert "net.2.bias" in updates

        # Verify local state was updated
        assert "net.0.weight" in executor.scaffold_c_local
