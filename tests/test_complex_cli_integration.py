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

from coreason_enclave.hardware.factory import get_attestation_provider
from coreason_enclave.hardware.real import RealAttestationProvider
from coreason_enclave.hardware.simulation import SimulationAttestationProvider
from coreason_enclave.main import main


def test_cli_integration_simulation_mode() -> None:
    """
    Verify that running main with --insecure actually causes the factory
    to return a SimulationAttestationProvider in the same process.
    """
    test_args = ["-w", "/tmp/ws", "-c", "conf.json", "--insecure"]

    # Ensure clean state
    with patch.dict(os.environ, {}, clear=True):
        # Run main
        # We don't need to mock logger, but we do need to suppress the actual NVFlare start
        # which is currently a 'pass', so it returns immediately.
        main(test_args)

        # Now verify that the factory logic, which reads the env var, works as expected.
        # This simulates the Executor calling the factory *after* main has set the env.
        provider = get_attestation_provider()
        assert isinstance(provider, SimulationAttestationProvider)


def test_cli_integration_secure_mode() -> None:
    """
    Verify that running main WITHOUT --insecure causes the factory
    to return a RealAttestationProvider.
    """
    test_args = ["-w", "/tmp/ws", "-c", "conf.json"]

    # Ensure clean state
    with patch.dict(os.environ, {}, clear=True):
        main(test_args)

        # Factory should return Real provider
        provider = get_attestation_provider()
        assert isinstance(provider, RealAttestationProvider)


def test_cli_integration_env_violation_abort() -> None:
    """
    Verify that if env is set to TRUE externally BUT flag is missing,
    the app aborts and does NOT initialize the provider (or crashes main).
    """
    test_args = ["-w", "/tmp/ws", "-c", "conf.json"]

    with patch.dict(os.environ, {"COREASON_ENCLAVE_SIMULATION": "true"}, clear=True):
        with pytest.raises(SystemExit):
            main(test_args)
