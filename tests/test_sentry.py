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

    def test_validate_input_success(self, sentry: DataSentry, mock_validator: MagicMock, tmp_path: Any) -> None:
        """Test validate_input with valid data and schema."""
        # Create a dummy data file
        data_file = tmp_path / "dataset.csv"
        data_file.touch()

        # Mock COREASON_DATA_ROOT
        with patch.dict(os.environ, {"COREASON_DATA_ROOT": str(tmp_path)}):
            mock_validator.validate.return_value = True
            result = sentry.validate_input("dataset.csv", schema={"type": "csv"})

            assert result is True
            mock_validator.validate.assert_called_once()
            # Check called path ends with dataset.csv
            args, _ = mock_validator.validate.call_args
            assert args[0].endswith("dataset.csv")

    def test_validate_input_file_not_found(self, sentry: DataSentry, tmp_path: Any) -> None:
        """Test validate_input raises FileNotFoundError for missing file."""
        with patch.dict(os.environ, {"COREASON_DATA_ROOT": str(tmp_path)}):
            with pytest.raises(FileNotFoundError):
                sentry.validate_input("non_existent_file.csv", schema={})

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

    def test_sanitize_output_valid(self, sentry: DataSentry) -> None:
        """Test sanitize_output with allowed keys."""
        payload = {"params": {"weights": [1, 2, 3]}, "metrics": {"loss": 0.5, "accuracy": 0.95}, "meta": {"round": 1}}
        result = sentry.sanitize_output(payload)
        assert result == payload

    def test_sanitize_output_invalid_key(self, sentry: DataSentry) -> None:
        """Test sanitize_output raises DataLeakageError for unauthorized keys."""
        payload = {
            "params": {},
            "raw_data": ["patient_record_1", "patient_record_2"],  # Unauthorized
        }
        with pytest.raises(DataLeakageError, match="Unauthorized output key detected: raw_data"):
            sentry.sanitize_output(payload)

    def test_sanitize_output_mixed_keys(self, sentry: DataSentry) -> None:
        """Test sanitize_output raises error even if some keys are valid."""
        payload = {"metrics": {}, "private_key": "secret"}
        with pytest.raises(DataLeakageError, match="Unauthorized output key detected: private_key"):
            sentry.sanitize_output(payload)
