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
Hardware Interfaces.

Defines the contract for hardware attestation providers.
"""

from abc import ABC, abstractmethod

from coreason_enclave.schemas import AttestationReport


class AttestationProvider(ABC):
    """
    Abstract Base Class for hardware attestation providers.

    Responsible for generating cryptographic evidence of the enclave's identity and integrity.
    """

    @abstractmethod
    def attest(self) -> AttestationReport:
        """
        Generate an attestation report for the current environment.

        Returns:
            AttestationReport: The signed report containing measurements and status.
        """
        pass  # pragma: no cover
