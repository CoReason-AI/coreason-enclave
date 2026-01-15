# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_enclave

from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from coreason_enclave.sentry import DataLeakageError, DataSentry, FileExistenceValidator


class TestDataSentry:
    @pytest.fixture
    def mock_validator(self) -> MagicMock:
        return MagicMock()

    @pytest.fixture
    def sentry(self, mock_validator: MagicMock) -> DataSentry:
        return DataSentry(validator=mock_validator)

    def test_init(self, sentry: DataSentry) -> None:
        assert sentry.validator is not None

    def test_validate_input_success(self, sentry: DataSentry, mock_validator: MagicMock) -> None:
        mock_validator.validate.return_value = True
        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.is_relative_to", return_value=True),
            patch("pathlib.Path.resolve") as mock_resolve,
        ):
            # Setup resolve to return a path object that supports is_relative_to
            mock_path = MagicMock()
            mock_path.is_relative_to.return_value = True
            mock_path.exists.return_value = True
            mock_resolve.return_value = mock_path

            assert sentry.validate_input("dataset_123", schema={}) is True
            mock_validator.validate.assert_called_once()

    def test_validate_input_resolve_error(self, sentry: DataSentry) -> None:
        # We need to simulate the resolve() call failing for the FULL path,
        # BUT NOT for the root path (which happens first on line 102).

        # We can mock Path object itself so that when it is initialized with root, it resolves fine,
        # but when initialized with data_root/dataset_id, it fails.

        with patch("pathlib.Path.resolve") as mock_resolve:
            # First call (root) succeeds
            # Second call (full path) fails
            mock_resolve.side_effect = [MagicMock(), RuntimeError("Resolution failed")]

            with pytest.raises(ValueError, match="Invalid dataset_id format"):
                sentry.validate_input("malformed_path", schema={})

    def test_validate_input_not_exists(self, sentry: DataSentry) -> None:
        with patch("pathlib.Path.resolve") as mock_resolve:
            mock_path = MagicMock()
            mock_path.is_relative_to.return_value = True
            mock_path.exists.return_value = False
            mock_resolve.return_value = mock_path

            with pytest.raises(FileNotFoundError):
                sentry.validate_input("missing_data", schema={})

    def test_validate_input_traversal(self, sentry: DataSentry) -> None:
        with patch("pathlib.Path.resolve") as mock_resolve:
            mock_path = MagicMock()
            mock_path.is_relative_to.return_value = False
            mock_resolve.return_value = mock_path

            with pytest.raises(ValueError, match="Path traversal"):
                sentry.validate_input("../secret", schema={})

    def test_validate_input_validator_returns_false(self, sentry: DataSentry, mock_validator: MagicMock) -> None:
        # Test case for validator returning False (lines 130-131)
        mock_validator.validate.return_value = False
        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.is_relative_to", return_value=True),
            patch("pathlib.Path.resolve") as mock_resolve,
        ):
            mock_path = MagicMock()
            mock_path.is_relative_to.return_value = True
            mock_path.exists.return_value = True
            mock_resolve.return_value = mock_path

            assert sentry.validate_input("dataset_123", schema={}) is False

    def test_validate_input_validator_raises(self, sentry: DataSentry, mock_validator: MagicMock) -> None:
        # Test case for validator raising exception (lines 132-134)
        mock_validator.validate.side_effect = Exception("Validator crashed")
        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.is_relative_to", return_value=True),
            patch("pathlib.Path.resolve") as mock_resolve,
        ):
            mock_path = MagicMock()
            mock_path.is_relative_to.return_value = True
            mock_path.exists.return_value = True
            mock_resolve.return_value = mock_path

            with pytest.raises(Exception, match="Validator crashed"):
                sentry.validate_input("dataset_123", schema={})

    def test_sanitize_output_success(self, sentry: DataSentry) -> None:
        payload = {"params": {"w": 1.0}, "metrics": {"acc": 0.9}}
        sanitized = sentry.sanitize_output(payload)
        assert sanitized == payload

    def test_sanitize_output_block_key(self, sentry: DataSentry) -> None:
        payload = {"secret_key": "123"}
        with pytest.raises(DataLeakageError, match="Unauthorized output key"):
            sentry.sanitize_output(payload)

    def test_sanitize_output_nested_sensitive(self, sentry: DataSentry) -> None:
        payload = {"params": {"nested": {"secret": "value"}}}
        with pytest.raises(DataLeakageError, match="Sensitive key detected"):
            sentry.sanitize_output(payload)

    def test_sanitize_output_recursion_error(self, sentry: DataSentry) -> None:
        # Test case for RecursionError (lines 168-169)
        recursive_dict: Dict[str, Any] = {}
        recursive_dict["self"] = recursive_dict
        payload = {"params": recursive_dict}
        with pytest.raises(DataLeakageError, match="Payload too deep or contains circular references"):
            sentry.sanitize_output(payload)

    def test_sanitize_recursive_list(self, sentry: DataSentry) -> None:
        # Test case for list recursion (line 187)
        payload = {"params": [1, 2, {"safe": "val"}]}
        sanitized = sentry.sanitize_output(payload)
        assert sanitized["params"][2]["safe"] == "val"


class TestFileExistenceValidator:
    def test_validate_exists(self) -> None:
        validator = FileExistenceValidator()
        with patch("pathlib.Path.exists", return_value=True):
            assert validator.validate("test_path", schema=None) is True

    def test_validate_not_exists(self) -> None:
        validator = FileExistenceValidator()
        with patch("pathlib.Path.exists", return_value=False):
            assert validator.validate("test_path", schema=None) is False
