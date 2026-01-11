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
from unittest.mock import MagicMock, mock_open, patch

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

    def test_attest_uniqueness(self) -> None:
        """Test that subsequent attestations return unique node IDs."""
        provider = SimulatedAttestationProvider()
        report1 = provider.attest()
        report2 = provider.attest()
        assert report1.node_id != report2.node_id


class TestRealProvider:
    @patch("pathlib.Path.exists")
    def test_init_raises_error_if_no_hardware(self, mock_exists: MagicMock) -> None:
        """Test that RealAttestationProvider raises RuntimeError if no TEE device is found."""
        mock_exists.return_value = False

        with pytest.raises(RuntimeError) as excinfo:
            RealAttestationProvider()

        assert "No TEE hardware detected" in str(excinfo.value)

    @patch("pathlib.Path.exists")
    @patch("builtins.open", new_callable=mock_open)
    def test_init_succeeds_with_hardware(self, mock_file: MagicMock, mock_exists: MagicMock) -> None:
        """Test that RealAttestationProvider initializes if TEE device is found and readable."""
        # Make one path exist
        mock_exists.side_effect = lambda: True

        provider = RealAttestationProvider()
        assert isinstance(provider, RealAttestationProvider)
        assert provider.device_path is not None

    @patch("pathlib.Path.exists")
    @patch("builtins.open", new_callable=mock_open)
    def test_init_raises_permission_error(self, mock_file: MagicMock, mock_exists: MagicMock) -> None:
        """Test that PermissionError is raised when device exists but is not readable."""
        mock_exists.return_value = True
        mock_file.side_effect = PermissionError("Access denied")

        with pytest.raises(PermissionError) as excinfo:
            RealAttestationProvider()

        assert "Permission denied" in str(excinfo.value)
        assert "Ensure the current user belongs to the appropriate group" in str(excinfo.value)

    @patch("pathlib.Path.exists")
    @patch("builtins.open", new_callable=mock_open)
    def test_init_skips_device_on_os_error(self, mock_file: MagicMock, mock_exists: MagicMock) -> None:
        """Test that RealAttestationProvider skips a device if OSError (not PermissionError) occurs."""
        # 3 devices exist
        # 1. OSError (should be skipped)
        # 2. Success (should be picked)
        # 3. Not reached
        mock_exists.side_effect = lambda: True

        # side_effect for open:
        # Call 1: OSError
        # Call 2: Success (mock file object)
        mock_file.side_effect = [OSError("I/O error"), MagicMock()]

        provider = RealAttestationProvider()

        assert provider.device_path is not None
        # Should have tried at least twice. Since TEE_DEVICES list iteration order is fixed,
        # it depends on how many iterations open is called.
        # We assume the first device in list triggered OSError, second succeeded.

    @patch("pathlib.Path.exists")
    @patch("builtins.open", new_callable=mock_open)
    def test_attest_raises_not_implemented(self, mock_file: MagicMock, mock_exists: MagicMock) -> None:
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

    @patch.dict(os.environ, {"COREASON_ENCLAVE_SIMULATION": "invalid_value"})
    @patch("coreason_enclave.hardware.real.RealAttestationProvider._check_hardware")
    def test_get_provider_invalid_env_var(self, mock_check: MagicMock) -> None:
        """Test that invalid env var values default to Real provider (secure default)."""
        mock_check.return_value = True

        provider = get_attestation_provider()
        assert isinstance(provider, RealAttestationProvider)
