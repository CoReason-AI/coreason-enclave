import pytest
from unittest.mock import MagicMock
from coreason_enclave.services import CoreasonEnclaveService, CoreasonEnclaveServiceAsync, EnclaveStatus
from coreason_identity.models import UserContext
from coreason_enclave.schemas import FederationJob, PrivacyConfig
from nvflare.apis.signal import Signal


@pytest.mark.asyncio
async def test_check_hardware_trust_sync_exception_coverage() -> None:
    """
    Directly test _check_hardware_trust_sync to ensure exception re-raise path is covered.
    """
    service = CoreasonEnclaveServiceAsync()
    service.attestation_provider = MagicMock()
    service.attestation_provider.attest.side_effect = Exception("Test Exception")

    with pytest.raises(Exception, match="Test Exception"):
        service._check_hardware_trust_sync()

    assert service.status == EnclaveStatus.ERROR


@pytest.mark.asyncio
async def test_evaluate_model_coverage() -> None:
    """
    Test evaluate_model to ensure lines 378 and 388 are covered.
    """
    service = CoreasonEnclaveServiceAsync()

    # UserContext required
    with pytest.raises(ValueError, match="UserContext is required"):
        # We pass None to verify validation logic
        await service.evaluate_model(None, MagicMock(), {}, Signal())

    # Return empty dict
    context = UserContext(
        user_id="u", username="u", email="e", permissions=[],
        project_context="p", privacy_budget_spent=0.0, privacy_budget_limit=1.0
    )
    job_config = FederationJob(
        job_id="00000000-0000-0000-0000-000000000000",
        clients=["c"], min_clients=1, rounds=1,
        dataset_id="d", model_arch="m", strategy="FED_AVG",
        privacy=PrivacyConfig(noise_multiplier=1.0, max_grad_norm=1.0, target_epsilon=10.0),
        user_context=context,
    )

    result = await service.evaluate_model(context, job_config, {}, Signal())
    assert result == {}


def test_privacy_budget_initial_coverage() -> None:
    """
    Test get_privacy_budget when guard is None (Line 133).
    """
    service = CoreasonEnclaveServiceAsync()
    assert service.get_privacy_budget() == 0.0


def test_service_reentrant_context() -> None:
    """
    Test re-entrant context manager to cover lines 448 and 460.
    """
    service = CoreasonEnclaveService()

    # First entry
    with service:
        assert service._ref_count == 1
        assert service._portal is not None

        # Nested entry (Line 448)
        with service:
            assert service._ref_count == 2
            assert service._portal is not None

        # After nested exit (Line 460 should be hit internally)
        assert service._ref_count == 1
        assert service._portal is not None

    # Final exit
    assert service._ref_count == 0
    assert service._portal is None
