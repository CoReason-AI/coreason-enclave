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
from unittest.mock import MagicMock, patch

import coreason_enclave.utils.logger


def test_logger_initialization() -> None:
    """Test that the logger is initialized correctly and creates the log directory."""
    log_path = Path("logs")
    # Ensure it exists (it should have been created by import)
    assert log_path.exists()
    assert log_path.is_dir()


def test_logger_exports() -> None:
    """Test that logger is exported."""
    from coreason_enclave.utils.logger import logger

    assert logger is not None


def test_logger_creates_directory_if_missing() -> None:
    """Test that logger creation logic attempts to create directory if missing."""
    with patch("coreason_enclave.utils.logger.Path") as mock_path_cls:
        mock_path_instance = MagicMock()
        mock_path_cls.return_value = mock_path_instance
        # Simulate directory does not exist
        mock_path_instance.exists.return_value = False

        # Call the configuration function directly
        coreason_enclave.utils.logger._configure_logger()

        # Verify mkdir was called
        mock_path_instance.mkdir.assert_called_with(parents=True, exist_ok=True)
