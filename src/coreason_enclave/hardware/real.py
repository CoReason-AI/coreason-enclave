# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_enclave

from pathlib import Path
from typing import Optional

from coreason_enclave.schemas import AttestationReport
from coreason_enclave.utils.logger import logger

from .interface import AttestationProvider


class RealAttestationProvider(AttestationProvider):
    """
    Real provider that interacts with TEE hardware (SGX/SEV/TDX).
    """

    # Known device paths for TEEs
    TEE_DEVICES = [
        Path("/dev/sgx_enclave"),  # Intel SGX
        Path("/dev/sgx/enclave"),  # Intel SGX (alternate)
        Path("/dev/sev"),  # AMD SEV
        Path("/dev/sev-guest"),  # AMD SEV-SNP
        Path("/dev/tdx-guest"),  # Intel TDX
    ]

    def __init__(self) -> None:
        logger.info("Initializing Real Attestation Provider. Verifying hardware...")
        self.device_path: Optional[Path] = None
        self.device_path = self._check_hardware()

        if not self.device_path:
            error_msg = (
                "No TEE hardware detected! "
                "Ensure you are running on an SGX/SEV/TDX enabled machine "
                "or use COREASON_ENCLAVE_SIMULATION=true."
            )
            logger.critical(error_msg)
            raise RuntimeError(error_msg)

    def _check_hardware(self) -> Optional[Path]:
        """
        Check if any TEE device exists and is readable.
        Returns the path of the first valid device found.
        """
        for device in self.TEE_DEVICES:
            if device.exists():
                try:
                    # Attempt to open the device to verify permissions
                    with open(device, "rb"):
                        pass
                    logger.info(f"TEE Hardware detected and verified: {device}")
                    return device
                except PermissionError as e:
                    logger.error(f"Permission denied accessing TEE device: {device}")
                    raise PermissionError(
                        f"Permission denied: {device}. "
                        "Ensure the current user belongs to the appropriate group (e.g., 'sgx', 'kvm')."
                    ) from e
                except OSError as e:
                    logger.warning(f"Could not access {device}: {e}")
                    continue
        return None

    def attest(self) -> AttestationReport:
        """
        Generate a real attestation report.
        """
        # TODO: Implement actual ioctl calls to the driver to get the quote.
        # For now, we raise NotImplementedError as the logic requires C extensions or specific ioctl wrappers.
        raise NotImplementedError("Real hardware attestation not yet implemented.")
