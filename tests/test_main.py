# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_enclave

from unittest.mock import patch

import pytest

from coreason_enclave.main import main


def test_main_arguments() -> None:
    """Test that main parses arguments correctly and sets up the environment."""
    test_args = ["-w", "/tmp/workspace", "-c", "config.json"]

    # We mock ClientTrain or whatever internal NVFlare component we invoke
    # Since the current implementation just passes, we verify it runs without error
    # and logs the correct info.

    with patch("coreason_enclave.main.logger") as mock_logger:
        main(test_args)

        # Verify logs
        mock_logger.info.assert_any_call("Starting Coreason Enclave Agent...")
        mock_logger.info.assert_any_call("Workspace: /tmp/workspace")
        mock_logger.info.assert_any_call("Config: config.json")
        mock_logger.info.assert_any_call("Invoking NVFlare ClientTrain...")


def test_main_missing_args() -> None:
    """Test that main raises SystemExit if required args are missing."""
    with pytest.raises(SystemExit):
        main([])  # Missing -w and -c
