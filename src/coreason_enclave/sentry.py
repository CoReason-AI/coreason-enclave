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
from pathlib import Path
from typing import Any, Dict, Protocol, runtime_checkable

from coreason_enclave.utils.logger import logger


@runtime_checkable
class ValidatorProtocol(Protocol):
    """
    Protocol for external data validators (e.g., coreason-validator).
    Decouples the enclave from specific validation implementations.
    """

    def validate(self, data_path: str, schema: Any) -> bool:
        """
        Validate the data at the given path against the provided schema.

        Args:
            data_path: Path to the data file or directory.
            schema: The schema definition (format depends on implementation).

        Returns:
            bool: True if valid, False otherwise.

        Raises:
            ValueError: If validation fails with specific errors.
        """
        ...


class DataLeakageError(Exception):
    """Raised when potential data leakage is detected in the output."""

    pass


class DataSentry:
    """
    The Firewall / Airlock for the Enclave.
    Responsible for input validation and output sanitation.
    """

    def __init__(self, validator: ValidatorProtocol) -> None:
        """
        Initialize the DataSentry.

        Args:
            validator: An implementation of ValidatorProtocol.
        """
        self.validator = validator
        self.allowed_output_keys = {"params", "metrics", "meta"}
        logger.info("DataSentry initialized.")

    def validate_input(self, dataset_id: str, schema: Any) -> bool:
        """
        Validate input data before it is used for training.

        Args:
            dataset_id: The identifier of the dataset (relative to COREASON_DATA_ROOT).
            schema: The expected schema of the data.

        Returns:
            bool: True if validation passes.

        Raises:
            FileNotFoundError: If the data path does not exist.
            ValueError: If validation fails.
        """
        data_root = os.environ.get("COREASON_DATA_ROOT", ".")
        data_path = Path(data_root) / dataset_id

        logger.info(f"Validating input data at: {data_path}")

        if not data_path.exists():
            logger.error(f"Data path not found: {data_path}")
            raise FileNotFoundError(f"Data path not found: {data_path}")

        # Delegate to the external validator
        try:
            is_valid = self.validator.validate(str(data_path), schema)
            if not is_valid:
                logger.error(f"Validation failed for dataset: {dataset_id}")
                return False
        except Exception as e:
            logger.exception(f"Exception during validation for {dataset_id}: {e}")
            raise

        logger.info(f"Input data {dataset_id} validated successfully.")
        return True

    def sanitize_output(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize the output payload ("Airlock").
        Ensures only allowed keys and types exit the enclave.

        Args:
            payload: The dictionary to be sent out of the enclave.

        Returns:
            Dict[str, Any]: The sanitized payload.

        Raises:
            DataLeakageError: If unauthorized keys or potential leaks are detected.
        """
        logger.info("Sanitizing output payload...")

        sanitized: Dict[str, Any] = {}

        for key, value in payload.items():
            if key not in self.allowed_output_keys:
                logger.warning(f"Blocking unauthorized output key: {key}")
                raise DataLeakageError(f"Unauthorized output key detected: {key}")

            # Deep inspection could go here (e.g., checking for PII in strings)
            # For now, we pass the allowed keys.
            sanitized[key] = value

        return sanitized
