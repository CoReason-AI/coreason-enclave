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
from unittest.mock import MagicMock, patch

import pytest

from coreason_enclave.federation.executor import CoreasonExecutor
from coreason_enclave.hardware.factory import get_attestation_provider
from coreason_enclave.hardware.real import RealAttestationProvider
from coreason_enclave.hardware.simulation import SimulationAttestationProvider
from coreason_enclave.main import apply_security_policy, main


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


class TestCLI:
    """Comprehensive tests for CLI argument parsing, security configuration, and edge cases."""

    @pytest.fixture
    def mock_sys_exit(self) -> Generator[Any, None, None]:
        with patch("sys.exit") as mock:
            yield mock

    @pytest.fixture
    def mock_client_train(self) -> Generator[MagicMock, None, None]:
        # Mock the new import location for client_train
        with patch("coreason_enclave.main.client_train") as mock:
            yield mock

    def test_apply_security_policy_simulation_flag(self) -> None:
        """Test that flags enable simulation mode."""
        with patch.dict(os.environ, {}, clear=True):
            # Case 1: --simulation=True, --insecure=False
            apply_security_policy(simulation_flag=True, insecure_flag=False)
            assert os.environ["COREASON_ENCLAVE_SIMULATION"] == "true"

    def test_apply_security_policy_insecure_flag(self) -> None:
        """Test that insecure flag enables simulation mode."""
        with patch.dict(os.environ, {}, clear=True):
            # Case 2: --simulation=False, --insecure=True
            apply_security_policy(simulation_flag=False, insecure_flag=True)
            assert os.environ["COREASON_ENCLAVE_SIMULATION"] == "true"

    def test_apply_security_policy_both_flags(self) -> None:
        """
        Edge Case: User passes both --simulation and --insecure.
        Should safely enable simulation mode without conflict.
        """
        with patch.dict(os.environ, {}, clear=True):
            apply_security_policy(simulation_flag=True, insecure_flag=True)
            assert os.environ["COREASON_ENCLAVE_SIMULATION"] == "true"

    def test_apply_security_policy_default(self) -> None:
        """Test that no flags enforces secure mode."""
        with patch.dict(os.environ, {}, clear=True):
            apply_security_policy(simulation_flag=False, insecure_flag=False)
            assert os.environ["COREASON_ENCLAVE_SIMULATION"] == "false"

    def test_apply_security_policy_env_precedence(self) -> None:
        """
        Edge Case: Environment says false/secure, but Flag says true/simulation.
        Flag should override environment to true.
        """
        with patch.dict(os.environ, {"COREASON_ENCLAVE_SIMULATION": "false"}):
            apply_security_policy(simulation_flag=True, insecure_flag=False)
            assert os.environ["COREASON_ENCLAVE_SIMULATION"] == "true"

    def test_apply_security_policy_garbage_env(self) -> None:
        """
        Edge Case: Environment variable contains garbage.
        Should default to False (Secure) and not crash or enable simulation.
        """
        with patch.dict(os.environ, {"COREASON_ENCLAVE_SIMULATION": "garbage_value"}):
            apply_security_policy(simulation_flag=False, insecure_flag=False)
            assert os.environ["COREASON_ENCLAVE_SIMULATION"] == "false"

    def test_apply_security_policy_violation(self) -> None:
        """Test that env=true without flags raises RuntimeError."""
        with patch.dict(os.environ, {"COREASON_ENCLAVE_SIMULATION": "true"}):
            with pytest.raises(RuntimeError, match="required '--insecure' or '--simulation' CLI flag is missing"):
                apply_security_policy(simulation_flag=False, insecure_flag=False)

    def test_main_arg_parsing_simulation(self, mock_client_train: MagicMock) -> None:
        """Test that main parses --simulation correctly."""
        with patch("coreason_enclave.main.apply_security_policy") as mock_config:
            main(["--workspace", "/tmp", "--conf", "config.json", "--simulation"])
            mock_config.assert_called_once_with(simulation_flag=True, insecure_flag=False)

    def test_main_arg_parsing_insecure(self, mock_client_train: MagicMock) -> None:
        """Test that main parses --insecure correctly."""
        with patch("coreason_enclave.main.apply_security_policy") as mock_config:
            main(["--workspace", "/tmp", "--conf", "config.json", "--insecure"])
            mock_config.assert_called_once_with(simulation_flag=False, insecure_flag=True)

    def test_main_required_args_missing(self) -> None:
        """
        Edge Case: Arguments required by NVFlare (workspace, conf) are missing.
        Even with --simulation, main() should exit/fail due to missing args.
        """
        with patch("sys.stderr", new_callable=MagicMock):
            with pytest.raises(SystemExit) as excinfo:
                main(["--simulation"])
            assert excinfo.value.code != 0

    def test_end_to_end_simulation_flow(self, mock_client_train: MagicMock) -> None:
        """
        Complex Scenario:
        1. Simulate CLI invocation with --simulation.
        2. Verify security mode is configured.
        3. Verify Factory produces Simulation provider.
        4. Verify Executor passes hardware trust check.
        """
        with patch.dict(os.environ, {}, clear=True):
            test_args = ["--workspace", "/tmp", "--conf", "config.json", "--simulation"]

            # Execute main
            main(test_args)

            # Verify Environment State
            assert os.environ.get("COREASON_ENCLAVE_SIMULATION") == "true"

            # Verify Factory Behavior
            provider = get_attestation_provider()
            assert isinstance(provider, SimulationAttestationProvider)

            # Verify Executor Behavior
            executor = CoreasonExecutor()
            assert isinstance(executor.attestation_provider, SimulationAttestationProvider)
            executor._check_hardware_trust()

            report = executor.attestation_provider.attest()
            assert report.status == "TRUSTED"
