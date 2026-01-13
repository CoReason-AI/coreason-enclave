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
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from coreason_enclave.sentry import DataLeakageError, DataSentry, ValidatorProtocol


class TestDataSentry:
    @pytest.fixture
    def mock_validator(self) -> MagicMock:
        """Fixture for a mocked ValidatorProtocol."""
        validator = MagicMock(spec=ValidatorProtocol)
        return validator

    @pytest.fixture
    def sentry(self, mock_validator: MagicMock) -> DataSentry:
        """Fixture for DataSentry instance."""
        return DataSentry(validator=mock_validator)

    def test_init(self, sentry: DataSentry) -> None:
        """Test initialization of DataSentry."""
        assert sentry is not None
        assert isinstance(sentry.validator, ValidatorProtocol)

    # --- validate_input tests ---

    def test_validate_input_success(self, sentry: DataSentry, mock_validator: MagicMock, tmp_path: Any) -> None:
        """Test validate_input with valid data and schema."""
        data_file = tmp_path / "dataset.csv"
        data_file.touch()

        with patch.dict(os.environ, {"COREASON_DATA_ROOT": str(tmp_path)}):
            mock_validator.validate.return_value = True
            result = sentry.validate_input("dataset.csv", schema={"type": "csv"})

            assert result is True
            mock_validator.validate.assert_called_once()
            args, _ = mock_validator.validate.call_args
            assert args[0].endswith("dataset.csv")

    def test_validate_input_file_not_found(self, sentry: DataSentry, tmp_path: Any) -> None:
        """Test validate_input raises FileNotFoundError for missing file."""
        with patch.dict(os.environ, {"COREASON_DATA_ROOT": str(tmp_path)}):
            with pytest.raises(FileNotFoundError):
                sentry.validate_input("non_existent_file.csv", schema={})

    def test_validate_input_path_traversal(self, sentry: DataSentry, tmp_path: Any) -> None:
        """
        Test that path traversal attempts are blocked.
        Trying to access a file outside COREASON_DATA_ROOT.
        """
        # Setup:
        # /tmp/allowed_root/
        # /tmp/secret_outside.txt
        root_dir = tmp_path / "allowed_root"
        root_dir.mkdir()

        outside_file = tmp_path / "secret_outside.txt"
        outside_file.touch()

        with patch.dict(os.environ, {"COREASON_DATA_ROOT": str(root_dir)}):
            # Attempt traversal: ../secret_outside.txt
            traversal_path = "../secret_outside.txt"

            with pytest.raises(ValueError, match="Path traversal detected"):
                sentry.validate_input(traversal_path, schema={})

    def test_validate_input_absolute_path_outside_root(self, sentry: DataSentry, tmp_path: Any) -> None:
        """Test that absolute paths pointing outside root are blocked."""
        root_dir = tmp_path / "allowed_root"
        root_dir.mkdir()

        outside_file = tmp_path / "secret.txt"
        outside_file.touch()

        with patch.dict(os.environ, {"COREASON_DATA_ROOT": str(root_dir)}):
            # Pass absolute path to outside file
            # Note: pathlib / operator behaves differently if 2nd arg is absolute on some OS/versions
            # behavior: Path('/a') / '/b' -> Path('/b')
            # So this simulates an attacker providing an absolute path.

            with pytest.raises(ValueError, match="Path traversal detected"):
                sentry.validate_input(str(outside_file), schema={})

    def test_validate_input_validation_failure(
        self, sentry: DataSentry, mock_validator: MagicMock, tmp_path: Any
    ) -> None:
        """Test validate_input returns False when validator fails."""
        data_file = tmp_path / "dataset.csv"
        data_file.touch()

        with patch.dict(os.environ, {"COREASON_DATA_ROOT": str(tmp_path)}):
            mock_validator.validate.return_value = False
            result = sentry.validate_input("dataset.csv", schema={})
            assert result is False

    def test_validate_input_validator_exception(
        self, sentry: DataSentry, mock_validator: MagicMock, tmp_path: Any
    ) -> None:
        """Test validate_input re-raises exception from validator."""
        data_file = tmp_path / "dataset.csv"
        data_file.touch()

        with patch.dict(os.environ, {"COREASON_DATA_ROOT": str(tmp_path)}):
            mock_validator.validate.side_effect = ValueError("Invalid schema")
            with pytest.raises(ValueError, match="Invalid schema"):
                sentry.validate_input("dataset.csv", schema={})

    def test_validate_input_resolve_exception(self, sentry: DataSentry, tmp_path: Any) -> None:
        """Test exception handling during path resolution."""
        # It's hard to trigger a resolve error with standard inputs, so we mock Path.resolve
        # We need two side effects:
        # 1. First call (root_path): Succeeds (returns tmp_path)
        # 2. Second call (full_path): Fails (raises OSError)

        def side_effect(*args: Any, **kwargs: Any) -> Any:
            # Check if we are resolving the root path or the dataset path
            # This is tricky because `self` in side_effect is the mock, not the Path object
            # But the mock replaces Path.resolve.
            # We can use an iterator.
            return next(effects)

        # Create an iterator for side effects
        effects = iter([tmp_path, OSError("Disk error")])

        with patch.dict(os.environ, {"COREASON_DATA_ROOT": str(tmp_path)}):
            with patch.object(Path, "resolve", side_effect=effects):
                with pytest.raises(ValueError, match="Invalid dataset_id format"):
                    sentry.validate_input("some_id", schema={})

    # --- sanitize_output tests ---

    def test_sanitize_output_valid(self, sentry: DataSentry) -> None:
        """Test sanitize_output with allowed keys."""
        payload = {"params": {"weights": [1, 2, 3]}, "metrics": {"loss": 0.5, "accuracy": 0.95}, "meta": {"round": 1}}
        result = sentry.sanitize_output(payload)
        assert result == payload

    def test_sanitize_output_invalid_top_level_key(self, sentry: DataSentry) -> None:
        """Test sanitize_output raises DataLeakageError for unauthorized keys."""
        payload = {
            "params": {},
            "raw_data": ["patient_record_1"],  # Unauthorized
        }
        with pytest.raises(DataLeakageError, match="Unauthorized output key detected: raw_data"):
            sentry.sanitize_output(payload)

    def test_sanitize_output_nested_sensitive_key(self, sentry: DataSentry) -> None:
        """
        Test Recursive Data Leakage check.
        A sensitive key (e.g. 'private_key') hidden inside a valid top-level key.
        """
        payload = {
            "meta": {
                "safe_info": "v1.0",
                "private_key": "MIIEvAIBADAN...",  # Malicious
            }
        }
        with pytest.raises(DataLeakageError, match="Sensitive key detected in nested output: private_key"):
            sentry.sanitize_output(payload)

    def test_sanitize_output_deeply_nested_sensitive_key(self, sentry: DataSentry) -> None:
        """Test deep recursion for sensitive keys."""
        payload = {
            "metrics": {
                "layer_stats": {
                    "conv1": {
                        "secret": "hidden_value"  # Malicious
                    }
                }
            }
        }
        with pytest.raises(DataLeakageError, match="Sensitive key detected in nested output: secret"):
            sentry.sanitize_output(payload)

    def test_sanitize_output_list_recursion(self, sentry: DataSentry) -> None:
        """Test that lists are also traversed."""
        payload = {
            "meta": [
                {"safe": 1},
                {"patient_id": 12345},  # Malicious
            ]
        }
        with pytest.raises(DataLeakageError, match="Sensitive key detected in nested output: patient_id"):
            sentry.sanitize_output(payload)

    def test_sanitize_output_empty(self, sentry: DataSentry) -> None:
        """Test empty payload."""
        assert sentry.sanitize_output({}) == {}
