# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_enclave

import importlib
import shutil
import sys
from pathlib import Path

# We need to import the actual module object, not the logger object inside it.
# coreason_enclave.utils.logger is the module.
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


def test_logger_exports() -> None:
    """Test that logger is exported."""
    assert logger is not None


def test_logger_directory_creation() -> None:
    """Test that the logger module creates the logs directory if it doesn't exist."""
    # Remove the logs directory if it exists
    log_path = Path("logs")
    if log_path.exists():
        shutil.rmtree(log_path)

    assert not log_path.exists()

    # We must explicitly reload the module found in sys.modules
    # 'logger_module' imported above might just be the object if the import machinery
    # confused it with the variable 'logger' (unlikely but possible if __init__ exports it).
    # Let's ensure we get the module from sys.modules

    module = sys.modules["coreason_enclave.utils.logger"]
    importlib.reload(module)

    # Verify directory was recreated
    assert log_path.exists()
    assert log_path.is_dir()
