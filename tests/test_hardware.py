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


def test_simulation_provider_uniqueness() -> None:
    """Test that subsequent calls to simulation provider generate unique IDs."""
    provider = SimulationAttestationProvider()
    report1 = provider.attest()
    report2 = provider.attest()

    assert report1.node_id != report2.node_id
    assert report1.enclave_signature != report2.enclave_signature


def test_factory_returns_simulation_provider_case_insensitive() -> None:
    """Test that the factory handles 'TRUE', 'True', 'true' correctly."""
    cases = ["true", "True", "TRUE", "TrUe"]
    for case in cases:
        with patch.dict(os.environ, {"COREASON_ENCLAVE_SIMULATION": case}):
            provider = get_attestation_provider()
            assert isinstance(provider, SimulationAttestationProvider), f"Failed for value: {case}"


def test_factory_defaults_to_real_on_invalid_simulation_flag() -> None:
    """Test that ambiguous values (yes, 1, on) default to Real mode for safety."""
    invalid_cases = ["yes", "1", "on", "enable", "false", "", "random"]
    for case in invalid_cases:
        with patch.dict(os.environ, {"COREASON_ENCLAVE_SIMULATION": case}):
            provider = get_attestation_provider()
            assert isinstance(provider, RealAttestationProvider), f"Failed for value: {case}"


def test_factory_returns_real_provider_by_default() -> None:
    """Test that the factory returns real provider when env var is NOT set."""
    # Ensure env var is not present
    with patch.dict(os.environ, {}, clear=True):
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


def test_real_provider_device_priority() -> None:
    """
    Test that RealAttestationProvider checks devices in order.
    We mock os.path.exists to fail the first one but pass the second.
    """
    provider = RealAttestationProvider()

    # Devices list in code: ["/dev/sgx_enclave", "/dev/sev", "/dev/tdx"]
    # We simulate sgx missing, but sev present.
    def mock_exists(path: str) -> bool:
        if path == "/dev/sgx_enclave":
            return False
        if path == "/dev/sev":
            return True
        return False

    with patch("os.path.exists", side_effect=mock_exists):
        # Should NOT raise RuntimeError (hardware not found)
        # Should raise NotImplementedError (because we found hardware but logic isn't done)
        with pytest.raises(NotImplementedError) as excinfo:
            provider.attest()
        # Verify it passed the hardware check
        assert "No TEE hardware detected" not in str(excinfo.value)
