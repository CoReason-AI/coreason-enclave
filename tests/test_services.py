# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_enclave

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from coreason_identity.models import UserContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal

from coreason_enclave.services import CoreasonEnclaveService, CoreasonEnclaveServiceAsync
from coreason_enclave.schemas import AggregationStrategy, FederationJob, PrivacyConfig


@pytest.mark.asyncio
class TestCoreasonEnclaveServiceAsync:
    @pytest.fixture
    def service(self) -> CoreasonEnclaveServiceAsync:
        svc = CoreasonEnclaveServiceAsync()
        return svc

    @pytest.fixture
    def context(self) -> UserContext:
        return UserContext(
            sub="test-user",
            email="test@coreason.ai",
            permissions=[],
            project_context="test",
        )

    async def test_lifecycle(self) -> None:
        async with CoreasonEnclaveServiceAsync() as svc:
            assert svc._client is not None
            # Internal client should be closed on exit (verified by logic, but hard to test directly without mock)

    async def test_check_hardware_trust(self, service: CoreasonEnclaveServiceAsync) -> None:
        service.attestation_provider = MagicMock()
        service.attestation_provider.attest.return_value.status = "TRUSTED"
        service.attestation_provider.attest.return_value.hardware_type = "SIMULATION"

        await service.check_hardware_trust()

        service.attestation_provider.attest.assert_called_once()

    async def test_execute_training_task_attestation_failure(
        self, service: CoreasonEnclaveServiceAsync, context: UserContext
    ) -> None:
        service.attestation_provider = MagicMock()
        service.attestation_provider.attest.return_value.status = "UNTRUSTED"

        shareable = Shareable()
        # Create a dummy config so validation passes before checking hardware
        dummy_config = FederationJob(
            job_id="00000000-0000-0000-0000-000000000000",
            clients=["c1"],
            min_clients=1,
            rounds=1,
            dataset_id="ds",
            model_arch="SimpleMLP",
            strategy=AggregationStrategy.FED_AVG,
            privacy=PrivacyConfig(noise_multiplier=1.0, max_grad_norm=1.0, target_epsilon=10.0),
        )
        shareable.set_header("job_config", dummy_config.model_dump_json())

        with pytest.raises(RuntimeError):
            await service.execute_training_task(shareable, Signal(), context=context)


class TestCoreasonEnclaveService:
    @pytest.fixture
    def context(self) -> UserContext:
        return UserContext(
            sub="test-user",
            email="test@coreason.ai",
            permissions=[],
            project_context="test",
        )

    def test_sync_facade_training(self, context: UserContext) -> None:
        # Mocking the async service inside the sync facade
        with patch("coreason_enclave.services.CoreasonEnclaveServiceAsync") as MockAsyncService:
            # Setup mock
            mock_async_instance = MockAsyncService.return_value
            mock_async_instance.__aenter__ = AsyncMock(return_value=mock_async_instance)
            mock_async_instance.__aexit__ = AsyncMock(return_value=None)
            mock_async_instance.execute_training_task = AsyncMock(return_value={"params": {}})

            service = CoreasonEnclaveService()

            with service:
                # Mock call
                shareable = Shareable()
                signal = Signal()
                result = service.execute_training_task(shareable, signal, context=context)

                assert result == {"params": {}}

    def test_service_outside_context(self, context: UserContext) -> None:
        """Test that using service outside context manager raises RuntimeError."""
        service = CoreasonEnclaveService()
        with pytest.raises(RuntimeError, match="Service used outside of context manager"):
            service.execute_training_task(Shareable(), Signal(), context=context)
