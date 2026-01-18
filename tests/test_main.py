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


@patch.dict(os.environ, {}, clear=True)
def test_main_insecure_mode() -> None:
    """Test that --insecure flag sets the simulation environment variable."""
    test_args = ["-w", "/tmp/ws", "-c", "conf.json", "--insecure"]

    with patch("coreason_enclave.main.logger") as mock_logger:
        main(test_args)

        assert os.environ.get("COREASON_ENCLAVE_SIMULATION") == "true"
        mock_logger.warning.assert_any_call("!!! RUNNING IN INSECURE SIMULATION MODE !!!")


@patch.dict(os.environ, {"COREASON_ENCLAVE_SIMULATION": "true"}, clear=True)
def test_main_aborts_on_env_mismatch() -> None:
    """Test that existing true env var WITHOUT flag causes abort."""
    test_args = ["-w", "/tmp/ws", "-c", "conf.json"]

    with patch("coreason_enclave.main.logger") as mock_logger:
        with pytest.raises(SystemExit):
            main(test_args)

        # Should log a critical error
        mock_logger.critical.assert_called_with(
            "Security Violation: COREASON_ENCLAVE_SIMULATION=true is set in the environment, "
            "but the required '--insecure' or '--simulation' CLI flag is missing. "
            "Refusing to launch in insecure mode without explicit CLI override."
        )


@patch.dict(os.environ, {}, clear=True)
def test_main_secure_default() -> None:
    """Test that without flag, simulation mode is NOT enabled."""
    test_args = ["-w", "/tmp/ws", "-c", "conf.json"]

    with patch("coreason_enclave.main.logger") as mock_logger:
        main(test_args)

        # Should be absent or false
        val = os.environ.get("COREASON_ENCLAVE_SIMULATION", "false")
        assert val != "true"
        mock_logger.info.assert_any_call("Running in SECURE HARDWARE MODE. TEE Attestation required.")


@patch.dict(os.environ, {"COREASON_ENCLAVE_SIMULATION": "false"}, clear=True)
def test_insecure_flag_overrides_env_false() -> None:
    """Test that --insecure flag overrides an explicit false env var."""
    test_args = ["-w", "/tmp/ws", "-c", "conf.json", "--insecure"]

    with patch("coreason_enclave.main.logger") as mock_logger:
        main(test_args)

        assert os.environ.get("COREASON_ENCLAVE_SIMULATION") == "true"
        mock_logger.warning.assert_any_call("!!! RUNNING IN INSECURE SIMULATION MODE !!!")


@patch.dict(os.environ, {"COREASON_ENCLAVE_SIMULATION": "garbage_value"}, clear=True)
def test_insecure_flag_overrides_garbage_env() -> None:
    """Test that --insecure flag overrides a garbage env var."""
    test_args = ["-w", "/tmp/ws", "-c", "conf.json", "--insecure"]

    with patch("coreason_enclave.main.logger") as mock_logger:
        main(test_args)

        assert os.environ.get("COREASON_ENCLAVE_SIMULATION") == "true"
        mock_logger.warning.assert_any_call("!!! RUNNING IN INSECURE SIMULATION MODE !!!")
