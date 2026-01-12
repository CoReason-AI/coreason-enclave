# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_enclave

import os
from unittest.mock import patch

import pytest

from coreason_enclave.hardware import (
    RealAttestationProvider,
    SimulationAttestationProvider,
    get_attestation_provider,
)
from coreason_enclave.schemas import AttestationReport


def test_simulation_provider_generates_valid_report() -> None:
    """Test that the simulation provider returns a valid AttestationReport."""
    provider = SimulationAttestationProvider()
    report = provider.attest()

    assert isinstance(report, AttestationReport)
    assert report.hardware_type == "SIMULATION_MODE"
    assert report.status == "TRUSTED"
    assert len(report.node_id) > 0
    assert len(report.measurement_hash) > 0
    assert len(report.enclave_signature) > 0


def test_factory_returns_simulation_provider_when_env_set() -> None:
    """Test that the factory returns simulation provider when env var is set."""
    with patch.dict(os.environ, {"COREASON_ENCLAVE_SIMULATION": "true"}):
        provider = get_attestation_provider()
        assert isinstance(provider, SimulationAttestationProvider)


def test_factory_returns_real_provider_by_default() -> None:
    """Test that the factory returns real provider when env var is NOT set."""
    # Ensure env var is not present
    with patch.dict(os.environ, {}, clear=True):
        # We need to preserve PATH and other base envs if necessary, but clearing SIMULATION is key.
        # However, clearing everything might break things if code relies on other envs.
        # Safer: ensure the specific key is missing.
        if "COREASON_ENCLAVE_SIMULATION" in os.environ:
            del os.environ["COREASON_ENCLAVE_SIMULATION"]

        provider = get_attestation_provider()
        assert isinstance(provider, RealAttestationProvider)


def test_real_provider_raises_error_without_hardware() -> None:
    """Test that RealAttestationProvider raises RuntimeError if /dev/sgx is missing."""
    provider = RealAttestationProvider()

    # Mock os.path.exists to return False for all devices
    with patch("os.path.exists", return_value=False):
        with pytest.raises(RuntimeError) as excinfo:
            provider.attest()
        assert "No TEE hardware detected" in str(excinfo.value)


def test_real_provider_raises_not_implemented_with_hardware() -> None:
    """Test that RealAttestationProvider raises NotImplementedError even if hardware 'exists' (for now)."""
    provider = RealAttestationProvider()

    # Mock os.path.exists to return True for one device
    with patch("os.path.exists", side_effect=lambda x: x == "/dev/sgx_enclave"):
        with pytest.raises(NotImplementedError) as excinfo:
            provider.attest()
        assert "not yet implemented" in str(excinfo.value)
