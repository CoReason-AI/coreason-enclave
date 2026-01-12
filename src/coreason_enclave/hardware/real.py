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

from coreason_enclave.hardware.interfaces import AttestationProvider
from coreason_enclave.schemas import AttestationReport


class RealAttestationProvider(AttestationProvider):
    """
    Real hardware attestation provider.
    Interacts with TEE drivers (e.g., /dev/sgx, /dev/sev) to generate quotes.
    """

    def attest(self) -> AttestationReport:
        """
        Generate a real attestation report.

        Raises:
            RuntimeError: If TEE hardware is not accessible.
        """
        # Preliminary check for common TEE devices
        # This is a basic check; implementation details depend on specific hardware (SGX/TDX/SEV)
        # and libraries (Gramine/SCONE)
        tee_devices = ["/dev/sgx_enclave", "/dev/sev", "/dev/tdx"]
        available_device = None

        for device in tee_devices:
            if os.path.exists(device):
                available_device = device
                break

        if not available_device:
            raise RuntimeError(
                "No TEE hardware detected. "
                "Ensure you are running on a Confidential Compute instance (SGX/SEV/TDX) "
                "or use simulation mode."
            )

        # In a real implementation, we would read the report from the driver or
        # call the Gramine/SCONE API here.
        raise NotImplementedError("Real hardware attestation is not yet implemented.")
