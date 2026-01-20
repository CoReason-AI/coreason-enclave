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
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal

from coreason_enclave.services import CoreasonEnclaveService, CoreasonEnclaveServiceAsync


@pytest.mark.asyncio
class TestCoreasonEnclaveServiceAsync:
    @pytest.fixture
    def service(self) -> CoreasonEnclaveServiceAsync:
        svc = CoreasonEnclaveServiceAsync()
        return svc

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

    async def test_execute_training_task_attestation_failure(self, service: CoreasonEnclaveServiceAsync) -> None:
        service.attestation_provider = MagicMock()
        service.attestation_provider.attest.return_value.status = "UNTRUSTED"

        with pytest.raises(RuntimeError):
            await service.execute_training_task(Shareable(), Signal())


class TestCoreasonEnclaveService:
    def test_sync_facade_training(self) -> None:
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
                result = service.execute_training_task(shareable, signal)

                assert result == {"params": {}}

            # Verification
            # Note: We can't easily verify anyio.run called __aenter__ directly on the mock class,
            # but we can check if methods were called.
            # However, since anyio.run runs in a loop, standard mocks might need care if checking call order.
            # But here we just check if it was called.
