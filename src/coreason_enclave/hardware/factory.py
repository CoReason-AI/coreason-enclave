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

from coreason_enclave.utils.logger import logger

from .interface import AttestationProvider
from .real import RealAttestationProvider
from .simulated import SimulatedAttestationProvider


def get_attestation_provider() -> AttestationProvider:
    """
    Factory function to return the appropriate AttestationProvider
    based on the COREASON_ENCLAVE_SIMULATION environment variable.
    """
    simulation_mode = os.environ.get("COREASON_ENCLAVE_SIMULATION", "false").lower() == "true"

    if simulation_mode:
        logger.info("Simulation mode enabled via COREASON_ENCLAVE_SIMULATION.")
        return SimulatedAttestationProvider()
    else:
        logger.info("Simulation mode disabled. Attempting to load Real Attestation Provider.")
        return RealAttestationProvider()
