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

from coreason_enclave.federation.executor import CoreasonExecutor
from coreason_enclave.hardware.factory import get_attestation_provider
from coreason_enclave.hardware.simulation import SimulationAttestationProvider
from coreason_enclave.main import configure_security_mode, main


class TestCLIEdgeCases:
    """Edge cases for CLI argument parsing and security configuration."""

    def test_both_flags_enabled(self) -> None:
        """
        Edge Case: User passes both --simulation and --insecure.
        Should safely enable simulation mode without conflict.
        """
        with patch.dict(os.environ, {}, clear=True):
            configure_security_mode(simulation_flag=True, insecure_flag=True)
            assert os.environ["COREASON_ENCLAVE_SIMULATION"] == "true"

    def test_env_false_flag_true(self) -> None:
        """
        Edge Case: Environment says false/secure, but Flag says true/simulation.
        Flag should override environment to true.
        """
        with patch.dict(os.environ, {"COREASON_ENCLAVE_SIMULATION": "false"}):
            configure_security_mode(simulation_flag=True, insecure_flag=False)
            assert os.environ["COREASON_ENCLAVE_SIMULATION"] == "true"

    def test_env_garbage_value(self) -> None:
        """
        Edge Case: Environment variable contains garbage.
        Should default to False (Secure) and not crash or enable simulation.
        """
        with patch.dict(os.environ, {"COREASON_ENCLAVE_SIMULATION": "garbage_value"}):
            configure_security_mode(simulation_flag=False, insecure_flag=False)
            assert os.environ["COREASON_ENCLAVE_SIMULATION"] == "false"

    def test_required_args_missing_with_simulation(self) -> None:
        """
        Edge Case: Arguments required by NVFlare (workspace, conf) are missing.
        Even with --simulation, main() should exit/fail due to missing args.
        """
        # We need to catch SystemExit(2) from argparse
        with patch("sys.stderr", new_callable=MagicMock):  # Suppress argparse error output
            with pytest.raises(SystemExit) as excinfo:
                main(["--simulation"])
            # Argparse usually exits with 2 for missing args
            assert excinfo.value.code != 0


class TestComplexScenarios:
    """Complex integration scenarios connecting CLI -> Config -> Factory -> Executor."""

    def test_end_to_end_simulation_flow(self) -> None:
        """
        Complex Scenario:
        1. Simulate CLI invocation with --simulation.
        2. Verify security mode is configured.
        3. Verify Factory produces Simulation provider.
        4. Verify Executor passes hardware trust check.
        """
        # We assume main() logic sets up the env.
        # We mock the actual NVFlare client execution part of main
        # to focus on the setup phase.

        # We need a clean env
        with patch.dict(os.environ, {}, clear=True):
            # Mock configure_security_mode to ensure it's actually called by main
            # But we also want its SIDE EFFECTS (setting env var).
            # So we wrap it instead of fully mocking it, or just trust the env var check.
            # Let's trust the env var check since we tested main->configure wiring in unit tests.

            # 1. Run Main with flags
            # We assume main sets the env var.
            # But wait, main() calls configure_security_mode.
            # Let's just call configure_security_mode directly to simulate that part of main,
            # OR run main() with a patch on the "sys.argv" replacement part so it doesn't actually run NVFlare.

            # Using main() is better for E2E.
            # We patch sys.argv reassignment or the code after argument parsing.

            test_args = ["--workspace", "/tmp", "--conf", "config.json", "--simulation"]

            # We patch the point where main would start NVFlare.
            # In existing main.py: it logs "Invoking NVFlare ClientTrain..." then passes.
            # So running main() is safe as long as we mock sys.argv/argparse if needed.

            # Actually, main() modifies sys.argv. We should ensure that doesn't break pytest execution flow
            # if we were running in same process, but here we are in a test function.

            with patch("coreason_enclave.main.sys.argv", ["original_script"]):
                main(test_args)

            # 2. Verify Environment State
            assert os.environ.get("COREASON_ENCLAVE_SIMULATION") == "true"

            # 3. Verify Factory Behavior
            provider = get_attestation_provider()
            assert isinstance(provider, SimulationAttestationProvider)

            # 4. Verify Executor Behavior
            # Instantiate Executor (it uses get_attestation_provider internally in __init__)
            executor = CoreasonExecutor()

            # Verify the internal provider is correct
            assert isinstance(executor.attestation_provider, SimulationAttestationProvider)

            # Verify the check_hardware_trust method (simulating what happens at start of task)
            # Should NOT raise RuntimeError
            executor._check_hardware_trust()

            # Verify report status
            report = executor.attestation_provider.attest()
            assert report.status == "TRUSTED"
            assert report.hardware_type == "SIMULATION_MODE"
