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
from unittest.mock import MagicMock, patch

import pytest

from coreason_enclave.hardware import get_attestation_provider
from coreason_enclave.hardware.real import RealAttestationProvider
from coreason_enclave.hardware.simulated import SimulatedAttestationProvider
from coreason_enclave.schemas import AttestationReport


class TestSimulatedProvider:
    def test_attest_returns_valid_report(self) -> None:
        """Test that the simulated provider returns a valid AttestationReport."""
        provider = SimulatedAttestationProvider()
        report = provider.attest()

        assert isinstance(report, AttestationReport)
        assert report.status == "TRUSTED"
        assert "SIMULATED_CPU" in report.hardware_type
        assert report.enclave_signature == "simulated_signature_insecure"


class TestRealProvider:
    @patch("pathlib.Path.exists")
    def test_init_raises_error_if_no_hardware(self, mock_exists: MagicMock) -> None:
        """Test that RealAttestationProvider raises RuntimeError if no TEE device is found."""
        mock_exists.return_value = False

        with pytest.raises(RuntimeError) as excinfo:
            RealAttestationProvider()

        assert "No TEE hardware detected" in str(excinfo.value)

    @patch("pathlib.Path.exists")
    def test_init_succeeds_with_hardware(self, mock_exists: MagicMock) -> None:
        """Test that RealAttestationProvider initializes if TEE device is found."""
        # Make one path exist
        mock_exists.side_effect = lambda: True

        provider = RealAttestationProvider()
        assert isinstance(provider, RealAttestationProvider)

    @patch("pathlib.Path.exists")
    def test_attest_raises_not_implemented(self, mock_exists: MagicMock) -> None:
        """Test that attest method is not yet implemented."""
        mock_exists.return_value = True
        provider = RealAttestationProvider()

        with pytest.raises(NotImplementedError):
            provider.attest()


class TestFactory:
    @patch.dict(os.environ, {"COREASON_ENCLAVE_SIMULATION": "true"})
    def test_get_provider_simulation_true(self) -> None:
        """Test that get_attestation_provider returns SimulatedAttestationProvider when env var is true."""
        provider = get_attestation_provider()
        assert isinstance(provider, SimulatedAttestationProvider)

    @patch.dict(os.environ, {"COREASON_ENCLAVE_SIMULATION": "false"})
    @patch("coreason_enclave.hardware.real.RealAttestationProvider._check_hardware")
    def test_get_provider_simulation_false(self, mock_check: MagicMock) -> None:
        """Test that get_attestation_provider returns RealAttestationProvider when env var is false."""
        # Mock hardware check to avoid RuntimeError
        mock_check.return_value = True

        provider = get_attestation_provider()
        assert isinstance(provider, RealAttestationProvider)

    @patch.dict(os.environ, {}, clear=True)
    @patch("coreason_enclave.hardware.real.RealAttestationProvider._check_hardware")
    def test_get_provider_default(self, mock_check: MagicMock) -> None:
        """Test that get_attestation_provider defaults to Real (simulation=false) if env var is missing."""
        # Mock hardware check
        mock_check.return_value = True

        provider = get_attestation_provider()
        assert isinstance(provider, RealAttestationProvider)
