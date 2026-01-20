# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_enclave

"""
Hardware Factory.

Provides the factory method to instantiate the correct AttestationProvider
based on the execution environment (Simulation vs. Real Hardware).
"""

import os

from coreason_enclave.hardware.interfaces import AttestationProvider
from coreason_enclave.hardware.real import RealAttestationProvider
from coreason_enclave.hardware.simulation import SimulationAttestationProvider
from coreason_enclave.utils.logger import logger


def get_attestation_provider() -> AttestationProvider:
    """
    Factory to return the appropriate AttestationProvider.

    Controlled by 'COREASON_ENCLAVE_SIMULATION' environment variable.
    If 'true', returns a simulation provider (for dev/test).
    Otherwise, returns the real hardware provider (requires TEE).

    Returns:
        AttestationProvider: An instance of SimulationAttestationProvider or RealAttestationProvider.
    """
    simulation_mode = os.getenv("COREASON_ENCLAVE_SIMULATION", "false").lower() == "true"

    if simulation_mode:
        logger.info("Initializing Simulation Attestation Provider.")
        return SimulationAttestationProvider()
    else:
        logger.info("Initializing Real Hardware Attestation Provider.")
        return RealAttestationProvider()
