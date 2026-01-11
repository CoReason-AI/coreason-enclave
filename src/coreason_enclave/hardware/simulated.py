# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_enclave

import hashlib
import platform
import uuid

from coreason_enclave.schemas import AttestationReport
from coreason_enclave.utils.logger import logger

from .interface import AttestationProvider


class SimulatedAttestationProvider(AttestationProvider):
    """
    Simulated provider for local development and testing.
    Does NOT provide real security guarantees.
    """

    def __init__(self) -> None:
        logger.warning("Initializing Simulated Attestation Provider. SECURITY: NONE.")

    def attest(self) -> AttestationReport:
        """
        Generate a dummy attestation report.
        """
        node_id = str(uuid.uuid4())
        # Simulate a hash of the binary
        measurement_hash = hashlib.sha256(b"simulated_binary").hexdigest()

        logger.info(f"Generating simulated attestation for node {node_id}")

        return AttestationReport(
            node_id=node_id,
            hardware_type=f"SIMULATED_CPU_{platform.processor()}",
            enclave_signature="simulated_signature_insecure",
            measurement_hash=measurement_hash,
            status="TRUSTED",
        )
