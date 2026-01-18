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
from typing import Any, Generator
from unittest.mock import patch

import pytest

from coreason_enclave.hardware.factory import get_attestation_provider
from coreason_enclave.hardware.real import RealAttestationProvider
from coreason_enclave.hardware.simulation import SimulationAttestationProvider
from coreason_enclave.main import configure_security_mode, main


class TestHardwareFactory:
    def test_get_provider_default(self) -> None:
        """Test that default environment returns RealAttestationProvider."""
        with patch.dict(os.environ, {}, clear=True):
            provider = get_attestation_provider()
            assert isinstance(provider, RealAttestationProvider)

    def test_get_provider_simulation(self) -> None:
        """Test that simulation environment returns SimulationAttestationProvider."""
        with patch.dict(os.environ, {"COREASON_ENCLAVE_SIMULATION": "true"}):
            provider = get_attestation_provider()
            assert isinstance(provider, SimulationAttestationProvider)

    def test_get_provider_simulation_case_insensitive(self) -> None:
        """Test that environment variable is case-insensitive."""
        with patch.dict(os.environ, {"COREASON_ENCLAVE_SIMULATION": "True"}):
            provider = get_attestation_provider()
            assert isinstance(provider, SimulationAttestationProvider)


class TestCLIEntry:
    @pytest.fixture
    def mock_sys_exit(self) -> Generator[Any, None, None]:
        with patch("sys.exit") as mock:
            yield mock

    def test_configure_security_mode_simulation_flag(self) -> None:
        """Test that flags enable simulation mode."""
        with patch.dict(os.environ, {}, clear=True):
            # Case 1: --simulation=True, --insecure=False
            configure_security_mode(simulation_flag=True, insecure_flag=False)
            assert os.environ["COREASON_ENCLAVE_SIMULATION"] == "true"

    def test_configure_security_mode_insecure_flag(self) -> None:
        """Test that insecure flag enables simulation mode."""
        with patch.dict(os.environ, {}, clear=True):
            # Case 2: --simulation=False, --insecure=True
            configure_security_mode(simulation_flag=False, insecure_flag=True)
            assert os.environ["COREASON_ENCLAVE_SIMULATION"] == "true"

    def test_configure_security_mode_default(self) -> None:
        """Test that no flags enforces secure mode."""
        with patch.dict(os.environ, {}, clear=True):
            configure_security_mode(simulation_flag=False, insecure_flag=False)
            assert os.environ["COREASON_ENCLAVE_SIMULATION"] == "false"

    def test_configure_security_mode_violation(self) -> None:
        """Test that env=true without flags raises RuntimeError."""
        with patch.dict(os.environ, {"COREASON_ENCLAVE_SIMULATION": "true"}):
            with pytest.raises(RuntimeError, match="required '--insecure' or '--simulation' CLI flag is missing"):
                configure_security_mode(simulation_flag=False, insecure_flag=False)

    def test_main_arg_parsing_simulation(self, mock_sys_exit: Any) -> None:
        """Test that main parses --simulation correctly."""
        # Mock configure_security_mode to verify it receives correct args
        with patch("coreason_enclave.main.configure_security_mode") as mock_config:
            # We don't actually run the client, just check parsing
            main(["--workspace", "/tmp", "--conf", "config.json", "--simulation"])

            # verify call
            mock_config.assert_called_once_with(simulation_flag=True, insecure_flag=False)

    def test_main_arg_parsing_insecure(self, mock_sys_exit: Any) -> None:
        """Test that main parses --insecure correctly."""
        with patch("coreason_enclave.main.configure_security_mode") as mock_config:
            main(["--workspace", "/tmp", "--conf", "config.json", "--insecure"])
            mock_config.assert_called_once_with(simulation_flag=False, insecure_flag=True)

    def test_main_no_flags(self, mock_sys_exit: Any) -> None:
        """Test that main parses no flags correctly."""
        with patch("coreason_enclave.main.configure_security_mode") as mock_config:
            main(["--workspace", "/tmp", "--conf", "config.json"])
            mock_config.assert_called_once_with(simulation_flag=False, insecure_flag=False)
