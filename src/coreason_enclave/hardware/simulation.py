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
Simulation Hardware Attestation.

Implementation of AttestationProvider for development and testing environments
where real TEE hardware is unavailable.
"""

import hashlib
from uuid import uuid4

from coreason_enclave.hardware.interfaces import AttestationProvider
from coreason_enclave.schemas import AttestationReport
from coreason_enclave.utils.logger import logger


class SimulationAttestationProvider(AttestationProvider):
    """
    Simulation provider for development and testing.

    Generates dummy attestation reports without requiring TEE hardware.
    WARNING: DO NOT USE IN PRODUCTION.
    """

    def attest(self) -> AttestationReport:
        """
        Generate a simulated attestation report.

        Returns:
            AttestationReport: A 'TRUSTED' report with mock signatures.
        """
        logger.warning("Generating SIMULATED attestation report. Do not use in production!")

        # Mock values
        node_id = str(uuid4())
        mock_binary_hash = hashlib.sha256(b"coreason-enclave-v0.1.0").hexdigest()
        mock_signature = "simulated_signature_" + hashlib.sha256(node_id.encode()).hexdigest()[:16]

        return AttestationReport(
            node_id=node_id,
            hardware_type="SIMULATION_MODE",
            enclave_signature=mock_signature,
            measurement_hash=mock_binary_hash,
            status="TRUSTED",
        )
