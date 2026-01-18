# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_enclave

from coreason_enclave.hardware.factory import get_attestation_provider
from coreason_enclave.hardware.interfaces import AttestationProvider
from coreason_enclave.hardware.real import RealAttestationProvider
from coreason_enclave.hardware.simulation import SimulationAttestationProvider

__all__ = [
    "AttestationProvider",
    "RealAttestationProvider",
    "SimulationAttestationProvider",
    "get_attestation_provider",
]
