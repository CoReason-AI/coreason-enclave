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

from coreason_enclave.main import main


def test_cli_integration_simulation_mode() -> None:
    """Test full CLI integration with simulation mode."""
    test_args = ["-w", "/tmp/ws", "-c", "conf.json", "--simulation"]

    # We mock client_train to avoid real NVFlare execution but verify our handoff
    with patch("coreason_enclave.main.client_train") as mock_client_train:
        # We also need to clear env to ensure fresh start
        with patch.dict(os.environ, {}, clear=True):
            main(test_args)

            # Assertions
            assert os.environ["COREASON_ENCLAVE_SIMULATION"] == "true"
            mock_client_train.parse_arguments.assert_called_once()
            mock_client_train.main.assert_called_once()


def test_cli_integration_secure_mode() -> None:
    """Test full CLI integration with secure mode (default)."""
    test_args = ["-w", "/tmp/ws", "-c", "conf.json"]

    with patch("coreason_enclave.main.client_train") as mock_client_train:
        with patch.dict(os.environ, {}, clear=True):
            main(test_args)

            # Assertions
            assert os.environ["COREASON_ENCLAVE_SIMULATION"] == "false"
            mock_client_train.parse_arguments.assert_called_once()
            mock_client_train.main.assert_called_once()


def test_main_missing_client_train_module() -> None:
    """Test fallback when client_train module is missing (ImportError simulation)."""
    test_args = ["-w", "/tmp/ws", "-c", "conf.json"]

    # Mock client_train as None to simulate ImportError handling
    with patch("coreason_enclave.main.client_train", None):
        with patch("coreason_enclave.main.logger") as mock_logger:
            with patch.dict(os.environ, {}, clear=True):
                main(test_args)

                # Should log warning
                mock_logger.warning.assert_called_with(
                    "NVFlare ClientTrain module not found. Skipping execution (Dry Run)."
                )
