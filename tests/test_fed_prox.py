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
from typing import Any, Dict, Generator
from unittest.mock import MagicMock, patch

import pytest
import torch
from coreason_identity.models import UserContext
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import ReturnCode, Shareable
from nvflare.apis.signal import Signal
from torch.utils.data import DataLoader

from coreason_enclave.federation import executor as executor_module
from coreason_enclave.federation.executor import CoreasonExecutor
from coreason_enclave.models.simple_mlp import SimpleMLP
from coreason_enclave.schemas import FederationJob

valid_user_context = UserContext(
    user_id="test_user", username="tester", privacy_budget_spent=0.0, privacy_budget_limit=10.0
)


class TestFedProxIntegration:
    @pytest.fixture
    def basic_job_config(self) -> Dict[str, Any]:
        return {
            "job_id": "123e4567-e89b-12d3-a456-426614174000",
            "clients": ["client1", "client2"],
            "min_clients": 2,
            "rounds": 5,
            "dataset_id": "test_data.csv",
            "model_arch": "SimpleMLP",
            "strategy": "FED_AVG",  # Default to FED_AVG
            "privacy": {"noise_multiplier": 1.0, "max_grad_norm": 1.0, "target_epsilon": 10.0},
            "user_context": {
                "user_id": "u1",
                "username": "user1",
                "privacy_budget_spent": 0.0,
                "privacy_budget_limit": 10.0,
            },
        }

    @pytest.fixture
    def mock_data_loader(self) -> DataLoader[Any]:
        # Use real DataLoader as Opacus inspects it deeply
        # Increase dataset size to avoid exploding epsilon (small dataset = high sampling rate)
        n_samples = 1000
        data = torch.randn(n_samples, 10)
        target = torch.randn(n_samples, 1)
        dataset = torch.utils.data.TensorDataset(data, target)
        # shuffle=False ensures deterministic batches for comparison
        return torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

    @pytest.fixture
    def context(self) -> UserContext:
        return UserContext(
            user_id="test-user",
            username="test-user",
            email="test@coreason.ai",
            permissions=[],
            project_context="test",
        )

    @pytest.fixture
    def executor(self, mock_data_loader: DataLoader[Any]) -> Generator[CoreasonExecutor, None, None]:
        # Patch dependencies before creating executor (which starts service)
        with (
            patch("coreason_enclave.services.get_attestation_provider") as mock_get_attestation,
            patch("coreason_enclave.services.DataSentry") as MockDataSentry,
            patch("coreason_enclave.services.DataLoaderFactory") as MockDataLoaderFactory,
        ):
            # Attestation
            provider = MagicMock()
            mock_get_attestation.return_value = provider
            provider.attest.return_value.status = "TRUSTED"
            provider.attest.return_value.hardware_type = "SIMULATION"

            # Sentry
            sentry = MockDataSentry.return_value
            sentry.sanitize_output.side_effect = lambda x: x

            # DataLoader
            loader_factory = MockDataLoaderFactory.return_value
            loader_factory.get_loader.return_value = mock_data_loader

            executor = CoreasonExecutor()

            # Attach mocks for verification if needed (not strictly used in current tests but good practice)
            # Access via private _async if we really need to inspect calls, but usually result code is enough.

            yield executor

    def test_fed_prox_vs_fed_avg(
        self,
        executor: CoreasonExecutor,
        basic_job_config: Dict[str, Any],
        mock_data_loader: DataLoader[Any],
        context: UserContext,
    ) -> None:
        """
        Test that FED_PROX produces a higher loss than FED_AVG given the same inputs
        and weights, because of the added proximal term.
        To observe this, we need the weights to move away from the global weights.
        """
        executor_module._CURRENT_CONTEXT = context

        # 1. Setup Common Inputs
        initial_params = SimpleMLP().state_dict()
        shareable_params = {k: v.clone() for k, v in initial_params.items()}

        # --- RUN 1: FED_AVG ---
        # Reset seed for Run 1
        torch.manual_seed(42)
        job_avg = basic_job_config.copy()
        job_avg["strategy"] = "FED_AVG"

        shareable_avg = Shareable()
        shareable_avg.set_header("job_config", json.dumps(job_avg))
        shareable_avg["params"] = shareable_params

        # Execute FedAvg
        result_avg = executor.execute(
            task_name="train", shareable=shareable_avg, fl_ctx=FLContext(), abort_signal=Signal()
        )

        loss_avg = result_avg.get("metrics")["loss"]

        # Verify weights moved in Avg run
        res_params_avg = result_avg["params"]
        # res_params_avg is dict of lists
        # keys might have _module prefix due to Opacus
        changed = False

        for k_init, v_init in initial_params.items():
            # Find corresponding key in result
            k_res = k_init
            if k_res not in res_params_avg and f"_module.{k_init}" in res_params_avg:
                k_res = f"_module.{k_init}"

            if k_res in res_params_avg:
                t_final = torch.tensor(res_params_avg[k_res])
                if not torch.allclose(v_init, t_final, atol=1e-6):
                    changed = True
                    break

        if not changed:
            pytest.fail("FED_AVG did not update weights! Optimizer or PrivacyGuard issue.")

        # --- RUN 2: FED_PROX ---
        # Reset seed for Run 2 to ensure identical data order and privacy noise
        torch.manual_seed(42)

        job_prox = basic_job_config.copy()
        job_prox["strategy"] = "FED_PROX"
        job_prox["proximal_mu"] = 1.0

        shareable_prox = Shareable()
        shareable_prox.set_header("job_config", json.dumps(job_prox))
        shareable_prox["params"] = shareable_params

        result_prox = executor.execute(
            task_name="train", shareable=shareable_prox, fl_ctx=FLContext(), abort_signal=Signal()
        )

        loss_prox = result_prox.get("metrics")["loss"]

        print(f"Loss Avg: {loss_avg}, Loss Prox: {loss_prox}")
        assert loss_prox > loss_avg

    def test_malformed_params_handling(
        self,
        executor: CoreasonExecutor,
        basic_job_config: Dict[str, Any],
        mock_data_loader: DataLoader[Any],
        context: UserContext,
    ) -> None:
        """Test that malformed incoming params are handled gracefully (warning logged)."""
        executor_module._CURRENT_CONTEXT = context
        # Ensure we don't crash when params are malformed
        torch.manual_seed(42)

        job = basic_job_config.copy()
        shareable = Shareable()
        shareable.set_header("job_config", json.dumps(job))

        # Malformed params: list instead of dict (executor expects dict for state_dict)
        # OR compatible dict but incompatible shapes
        shareable["params"] = {"invalid_layer": [1.0, 2.0]}
        # This will raise RuntimeError in load_state_dict because of strict=True (default)
        # or mismatch keys if we used strict=False (but we use default).

        # We just want to ensure it doesn't crash execution
        result = executor.execute(task_name="train", shareable=shareable, fl_ctx=FLContext(), abort_signal=Signal())

        assert result.get_return_code() == ReturnCode.EXECUTION_EXCEPTION  # Or Shareable default return code
        # Ideally we check logs, but verifying return code is OK implies exception was caught

    def test_proximal_mu_configuration(
        self, executor: CoreasonExecutor, basic_job_config: Dict[str, Any], mock_data_loader: DataLoader[Any]
    ) -> None:
        """Test that proximal_mu defaults correctly and can be overridden."""
        # 1. Default
        job = FederationJob(**basic_job_config)
        assert job.proximal_mu == 0.01

        # 2. Override
        basic_job_config["proximal_mu"] = 0.5
        job = FederationJob(**basic_job_config)
        assert job.proximal_mu == 0.5

        # 3. Invalid
        basic_job_config["proximal_mu"] = -0.1
        with pytest.raises(ValueError, match="non-negative"):
            FederationJob(**basic_job_config)

    def test_fed_prox_mu_zero(
        self,
        executor: CoreasonExecutor,
        basic_job_config: Dict[str, Any],
        mock_data_loader: DataLoader[Any],
        context: UserContext,
    ) -> None:
        """Test that setting proximal_mu=0 results in behavior identical to FED_AVG."""
        executor_module._CURRENT_CONTEXT = context
        # 1. Setup
        initial_params = SimpleMLP().state_dict()
        shareable_params = {k: v.clone() for k, v in initial_params.items()}

        # --- RUN 1: FED_AVG ---
        torch.manual_seed(42)
        job_avg = basic_job_config.copy()
        job_avg["strategy"] = "FED_AVG"
        shareable_avg = Shareable()
        shareable_avg.set_header("job_config", json.dumps(job_avg))
        shareable_avg["params"] = shareable_params

        result_avg = executor.execute(
            task_name="train", shareable=shareable_avg, fl_ctx=FLContext(), abort_signal=Signal()
        )
        loss_avg = result_avg.get("metrics")["loss"]

        # --- RUN 2: FED_PROX with mu=0 ---
        torch.manual_seed(42)
        job_prox_0 = basic_job_config.copy()
        job_prox_0["strategy"] = "FED_PROX"
        job_prox_0["proximal_mu"] = 0.0
        shareable_prox_0 = Shareable()
        shareable_prox_0.set_header("job_config", json.dumps(job_prox_0))
        shareable_prox_0["params"] = shareable_params

        result_prox_0 = executor.execute(
            task_name="train", shareable=shareable_prox_0, fl_ctx=FLContext(), abort_signal=Signal()
        )
        loss_prox_0 = result_prox_0.get("metrics")["loss"]

        # Expect identical loss
        assert loss_avg == pytest.approx(loss_prox_0, rel=1e-6)

    def test_fed_prox_no_incoming_params(
        self,
        executor: CoreasonExecutor,
        basic_job_config: Dict[str, Any],
        mock_data_loader: DataLoader[Any],
        context: UserContext,
    ) -> None:
        """Test that FedProx degrades gracefully (acts like FedAvg) if no global params are provided."""
        executor_module._CURRENT_CONTEXT = context
        # --- RUN 1: FED_AVG (No params provided) ---
        torch.manual_seed(42)
        job_avg = basic_job_config.copy()
        job_avg["strategy"] = "FED_AVG"
        shareable_avg = Shareable()
        shareable_avg.set_header("job_config", json.dumps(job_avg))
        # No params in shareable

        result_avg = executor.execute(
            task_name="train", shareable=shareable_avg, fl_ctx=FLContext(), abort_signal=Signal()
        )
        loss_avg = result_avg.get("metrics")["loss"]

        # --- RUN 2: FED_PROX (No params provided) ---
        torch.manual_seed(42)
        job_prox = basic_job_config.copy()
        job_prox["strategy"] = "FED_PROX"
        job_prox["proximal_mu"] = 1.0  # Should be ignored as no global params
        shareable_prox = Shareable()
        shareable_prox.set_header("job_config", json.dumps(job_prox))
        # No params in shareable

        result_prox = executor.execute(
            task_name="train", shareable=shareable_prox, fl_ctx=FLContext(), abort_signal=Signal()
        )
        loss_prox = result_prox.get("metrics")["loss"]

        # Expect identical loss because proximal term calculation is skipped
        assert loss_avg == pytest.approx(loss_prox, rel=1e-6)

    def test_fed_prox_frozen_layers(
        self,
        executor: CoreasonExecutor,
        basic_job_config: Dict[str, Any],
        mock_data_loader: DataLoader[Any],
        context: UserContext,
    ) -> None:
        """Test that FedProx ignores frozen layers (requires_grad=False)."""
        executor_module._CURRENT_CONTEXT = context

        # Define a model class with frozen layers
        class FrozenSimpleMLP(SimpleMLP):
            def __init__(self) -> None:
                super().__init__()
                # Freeze first layer
                for param in self.net[0].parameters():
                    param.requires_grad = False

        # Mock Registry to return this class
        with patch("coreason_enclave.models.registry.ModelRegistry.get", return_value=FrozenSimpleMLP):
            torch.manual_seed(42)
            job = basic_job_config.copy()
            job["strategy"] = "FED_PROX"
            job["proximal_mu"] = 1.0

            # Use standard params (they will be loaded into FrozenSimpleMLP)
            initial_params = SimpleMLP().state_dict()
            shareable_params = {k: v.clone() for k, v in initial_params.items()}

            shareable = Shareable()
            shareable.set_header("job_config", json.dumps(job))
            shareable["params"] = shareable_params

            # Run execution
            # This should trigger the 'if not param.requires_grad: continue' line
            result = executor.execute(task_name="train", shareable=shareable, fl_ctx=FLContext(), abort_signal=Signal())

            assert result.get_return_code() == ReturnCode.OK
