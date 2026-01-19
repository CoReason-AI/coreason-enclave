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
        ...  # pragma: no cover


class FileExistenceValidator:
    """
    A simple validator that checks for file existence.
    Satisfies ValidatorProtocol.
    """

    def validate(self, data_path: str, schema: Any) -> bool:
        """
        Validate that the file exists at the given path.
        Schema is ignored for this basic validation.
        """
        path = Path(data_path)
        if not path.exists():
            logger.error(f"Validation failed: File {data_path} does not exist.")
            return False
        logger.info(f"FileExistenceValidator: Validated existence of {data_path}")
        return True


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
        self.allowed_output_keys = {"params", "metrics", "meta", "scaffold_updates"}
        # Blocklist for sensitive keys that should NEVER appear, even nested.
        self.sensitive_keys = {"private_key", "secret", "patient_id", "raw_data", "pii"}
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
            ValueError: If validation fails or path traversal is detected.
        """
        data_root = os.environ.get("COREASON_DATA_ROOT", ".")
        root_path = Path(data_root).resolve()

        # Prevent Path Traversal
        # We join the path and resolve it to get the absolute path
        # Then we check if it starts with the root path
        try:
            full_path = (Path(data_root) / dataset_id).resolve()
        except Exception as e:
            # Handle edge cases where resolve fails or path is malformed
            logger.error(f"Failed to resolve path for {dataset_id}: {e}")
            raise ValueError(f"Invalid dataset_id format: {dataset_id}") from e

        # Secure check using pathlib's is_relative_to (Python 3.9+)
        # This prevents partial path traversal attacks (e.g. /data vs /database)
        if not full_path.is_relative_to(root_path):
            logger.error(f"Path traversal attempt detected! {dataset_id} -> {full_path}")
            raise ValueError(f"Invalid dataset_id: Path traversal detected for {dataset_id}")

        logger.info(f"Validating input data at: {full_path}")

        if not full_path.exists():
            logger.error(f"Data path not found: {full_path}")
            raise FileNotFoundError(f"Data path not found: {full_path}")

        # Delegate to the external validator
        try:
            is_valid = self.validator.validate(str(full_path), schema)
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
        Performs recursive checking to prevent nested data leakage.

        Args:
            payload: The dictionary to be sent out of the enclave.

        Returns:
            Dict[str, Any]: The sanitized payload.

        Raises:
            DataLeakageError: If unauthorized keys or potential leaks are detected.
        """
        logger.info("Sanitizing output payload...")

        sanitized: Dict[str, Any] = {}

        try:
            for key, value in payload.items():
                # 1. Top-level Allowlist Check
                if key not in self.allowed_output_keys:
                    logger.warning(f"Blocking unauthorized output key: {key}")
                    raise DataLeakageError(f"Unauthorized output key detected: {key}")

                # 2. Recursive Sensitivity Check
                sanitized[key] = self._sanitize_recursive(value)
        except RecursionError as e:
            logger.error(f"Recursion depth exceeded during sanitation: {e}")
            raise DataLeakageError("Payload too deep or contains circular references") from e

        return sanitized

    def _sanitize_recursive(self, value: Any) -> Any:
        """
        Recursively traverse the value to check for sensitive keys in nested dictionaries.
        """
        if isinstance(value, dict):
            clean_dict = {}
            for k, v in value.items():
                if k in self.sensitive_keys:
                    logger.critical(f"Sensitive key detected in nested output: {k}")
                    raise DataLeakageError(f"Sensitive key detected in nested output: {k}")
                clean_dict[k] = self._sanitize_recursive(v)
            return clean_dict

        elif isinstance(value, (list, tuple)):
            return [self._sanitize_recursive(v) for v in value]

        else:
            return value
