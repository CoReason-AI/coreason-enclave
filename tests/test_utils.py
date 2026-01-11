# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_enclave

from importlib import reload
from pathlib import Path
from unittest.mock import patch

from coreason_enclave.utils.logger import logger


def test_logger_initialization() -> None:
    """Test that the logger is initialized correctly and creates the log directory."""
    # Since the logger is initialized on import, we check side effects

    # Check if logs directory creation is handled
    # Note: running this test might actually create the directory in the test environment
    # if it doesn't exist.

    log_path = Path("logs")
    assert log_path.exists()
    assert log_path.is_dir()

    # Verify app.log creation if it was logged to (it might be empty or not created until log)
    # logger.info("Test log")
    # assert (log_path / "app.log").exists()


def test_logger_exports() -> None:
    """Test that logger is exported."""
    assert logger is not None


def test_logger_creates_logs_dir() -> None:
    """Test that logger creates logs directory if it doesn't exist."""
    # We use patch to simulate the directory not existing
    # We reload the module to trigger the top-level code again
    with (
        patch("pathlib.Path.exists", return_value=False),
        patch("pathlib.Path.mkdir") as mock_mkdir,
    ):
        import coreason_enclave.utils.logger

        reload(coreason_enclave.utils.logger)
        mock_mkdir.assert_called()
