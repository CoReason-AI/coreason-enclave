from typing import Generator
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from coreason_enclave.api import app
from coreason_enclave.schemas import AttestationReport
from coreason_enclave.services import CoreasonEnclaveService, EnclaveStatus


@pytest.fixture
def mock_service() -> Generator[CoreasonEnclaveService, None, None]:
    # Reset singleton
    CoreasonEnclaveService._instance = None

    with patch("coreason_enclave.services.get_attestation_provider") as mock_provider_factory:
        mock_provider = MagicMock()
        mock_report = AttestationReport(
            node_id="test-node",
            hardware_type="TEST_HARDWARE",
            enclave_signature="sig",
            measurement_hash="0" * 64,
            status="TRUSTED",
        )
        mock_provider.attest.return_value = mock_report
        mock_provider_factory.return_value = mock_provider

        service = CoreasonEnclaveService.get_instance()

        yield service

        # Cleanup
        if service._portal:
            service.__exit__(None, None, None)
        CoreasonEnclaveService._instance = None


def test_health_check(mock_service: CoreasonEnclaveService) -> None:
    # We use TestClient as context manager to trigger lifespan
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == EnclaveStatus.ATTESTED
        assert response.json()["trusted"] is True


def test_attestation_endpoint(mock_service: CoreasonEnclaveService) -> None:
    with TestClient(app) as client:
        response = client.get("/attestation")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "TRUSTED"
        assert data["hardware_type"] == "TEST_HARDWARE"


def test_privacy_budget_endpoint(mock_service: CoreasonEnclaveService) -> None:
    with TestClient(app) as client:
        # Mock privacy guard in service
        mock_service._async._current_privacy_guard = MagicMock()
        mock_service._async._current_privacy_guard.get_current_epsilon.return_value = 1.23

        response = client.get("/privacy/budget")
        assert response.status_code == 200
        assert response.json()["epsilon"] == 1.23


def test_startup_failure_untrusted() -> None:
    # Reset singleton to force fresh startup
    CoreasonEnclaveService._instance = None

    with patch("coreason_enclave.services.get_attestation_provider") as mock_provider_factory:
        mock_provider = MagicMock()
        # Mock UNTRUSTED report
        mock_report = AttestationReport(
            node_id="test-node",
            hardware_type="TEST_HARDWARE",
            enclave_signature="sig",
            measurement_hash="0" * 64,
            status="UNTRUSTED",
        )
        mock_provider.attest.return_value = mock_report
        mock_provider_factory.return_value = mock_provider

        # Expect RuntimeError during startup (lifespan)
        with pytest.raises(RuntimeError, match="Hardware not trusted"):
            with TestClient(app):
                pass


def test_attestation_endpoint_error(mock_service: CoreasonEnclaveService) -> None:
    # Start client first (lifespan succeeds)
    with TestClient(app) as client:
        # Now mock the method on the SINGLETON instance
        with patch.object(mock_service, "refresh_attestation", side_effect=Exception("Simulated Failure")):
            response = client.get("/attestation")
            assert response.status_code == 500
            assert response.json()["detail"] == "Attestation failed"


def test_health_check_initializing(mock_service: CoreasonEnclaveService) -> None:
    with TestClient(app) as client:
        # Force status AFTER startup
        with patch.object(mock_service._async, "status", EnclaveStatus.INITIALIZING):
            response = client.get("/health")
            assert response.status_code == 503
            assert response.json()["detail"] == "Enclave initializing"


def test_health_check_error_state(mock_service: CoreasonEnclaveService) -> None:
    with TestClient(app) as client:
        with patch.object(mock_service._async, "status", EnclaveStatus.ERROR):
            response = client.get("/health")
            assert response.status_code == 503
            assert response.json()["detail"] == "Enclave in ERROR state"
