# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_enclave

from typing import Any, Dict, Generator
from unittest.mock import MagicMock, patch

import pytest
import torch
from coreason_identity.models import UserContext
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import ReturnCode, Shareable
from nvflare.apis.signal import Signal

from coreason_enclave.federation import executor as executor_module
from coreason_enclave.federation.executor import CoreasonExecutor
from coreason_enclave.schemas import AggregationStrategy, FederationJob, PrivacyConfig

valid_user_context = UserContext(
    user_id="test_user", username="tester", privacy_budget_spent=0.0, privacy_budget_limit=10.0
)


class TestScaffoldIntegration:
    @pytest.fixture
    def context(self) -> UserContext:
        return UserContext(
            sub="test-user",
            email="test@coreason.ai",
            permissions=[],
            project_context="test",
        )

    @pytest.fixture
    def executor(self) -> Generator[CoreasonExecutor, None, None]:
        """Fixture for CoreasonExecutor."""
        with (
            patch("coreason_enclave.services.get_attestation_provider") as mock_get_attestation,
            patch("coreason_enclave.services.DataSentry") as MockDataSentry,
            patch("coreason_enclave.services.DataLoaderFactory") as MockDataLoaderFactory,
        ):
            # Attestation
            provider = MagicMock()
            mock_get_attestation.return_value = provider
            provider.attest.return_value.status = "TRUSTED"
            provider.attest.return_value.hardware_type = "TEST_HARDWARE"

            # Sentry
            sentry = MockDataSentry.return_value
            sentry.sanitize_output.side_effect = lambda x: x

            # DataLoader
            loader_factory = MockDataLoaderFactory.return_value
            dataset = torch.utils.data.TensorDataset(torch.randn(10, 10), torch.randn(10, 1))
            loader = torch.utils.data.DataLoader(dataset, batch_size=2)
            loader_factory.get_loader.return_value = loader

            executor = CoreasonExecutor()
            yield executor

    @pytest.fixture
    def job_config(self) -> FederationJob:
        return FederationJob(
            user_context=valid_user_context,
            job_id="00000000-0000-0000-0000-000000000000",
            clients=["client1"],
            min_clients=1,
            rounds=10,
            dataset_id="test_dataset",
            model_arch="SimpleMLP",
            strategy=AggregationStrategy.SCAFFOLD,
            privacy=PrivacyConfig(noise_multiplier=1.0, max_grad_norm=1.0, target_epsilon=10.0),
        )

    def test_scaffold_multi_round_persistence(
        self, executor: CoreasonExecutor, job_config: FederationJob, context: UserContext
    ) -> None:
        """Test complex scenario: Multiple rounds to verify state persistence."""
        executor_module._CURRENT_CONTEXT = context

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
        local_weight_r1 = executor.service._async.scaffold_c_local["net.0.weight"].clone()
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
        local_weight_r2 = executor.service._async.scaffold_c_local["net.0.weight"]
        assert not torch.allclose(local_weight_r2, local_weight_r1)

    def test_integration_execution(
        self, executor: CoreasonExecutor, job_config: FederationJob, context: UserContext
    ) -> None:
        """Test full execution flow with SCAFFOLD strategy."""
        executor_module._CURRENT_CONTEXT = context

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
        assert "net.0.weight" in executor.service._async.scaffold_c_local

    def test_scaffold_strategy_isolation(
        self, executor: CoreasonExecutor, job_config: FederationJob, context: UserContext
    ) -> None:
        """Test that other strategies (e.g. FED_AVG) do NOT affect SCAFFOLD state."""
        executor_module._CURRENT_CONTEXT = context

        # Modify job config to use FED_AVG
        job_config.strategy = AggregationStrategy.FED_AVG
        shareable = Shareable()
        shareable.set_header("job_config", job_config.model_dump_json())

        params = {
            "net.0.weight": torch.randn(16, 10),
            "net.0.bias": torch.randn(16),
            "net.2.weight": torch.randn(1, 16),
            "net.2.bias": torch.randn(1),
        }
        shareable["params"] = {k: v.numpy() for k, v in params.items()}

        # Even if scaffold globals are provided (should be ignored by strategy check)
        scaffold_c_global = {k: torch.zeros_like(v).numpy() for k, v in params.items()}
        shareable["scaffold_c_global"] = scaffold_c_global

        fl_ctx = FLContext()
        abort_signal = Signal()

        # Execute
        result = executor.execute("train", shareable, fl_ctx, abort_signal)

        assert result.get_return_code() == ReturnCode.OK

        # Verify NO scaffold updates in output
        assert "scaffold_updates" not in result

        # Verify local state is EMPTY (assuming fresh executor)
        assert len(executor.service._async.scaffold_c_local) == 0
