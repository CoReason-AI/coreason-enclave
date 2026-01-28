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

    # We mock start_client to avoid real NVFlare execution but verify our handoff
    with patch("coreason_enclave.main.start_client") as mock_start_client:
        # We also need to clear env to ensure fresh start
        with patch.dict(os.environ, {}, clear=True):
            main(test_args)

            # Assertions
            assert os.environ["COREASON_ENCLAVE_SIMULATION"] == "true"
            mock_start_client.assert_called_once()


def test_cli_integration_secure_mode() -> None:
    """Test full CLI integration with secure mode (default)."""
    test_args = ["-w", "/tmp/ws", "-c", "conf.json"]

    with patch("coreason_enclave.main.start_client") as mock_start_client:
        with patch.dict(os.environ, {}, clear=True):
            main(test_args)

            # Assertions
            assert os.environ["COREASON_ENCLAVE_SIMULATION"] == "false"
            mock_start_client.assert_called_once()


def test_main_missing_client_train_module() -> None:
    """Test fallback when client_train module is missing (ImportError simulation)."""
    # This logic moved to executor.start_client, so testing it via main requires mocking
    # executor.client_train to None
    test_args = ["-w", "/tmp/ws", "-c", "conf.json"]

    with patch("coreason_enclave.federation.executor.client_train", None):
        with patch("coreason_enclave.federation.executor.logger") as mock_logger:
            # We also need to prevent sys.argv backup crash if main invokes start_client which invokes logic
            # Actually, main invokes start_client.
            # And start_client checks `if client_train:`.
            # So we don't mock start_client here, we mock the underlying client_train in executor.
            # But wait, start_client is imported in main.
            # We are testing main -> start_client -> fallback.

            # We need to mock start_client? No, we want to test the fallback logic inside start_client?
            # Or main's handling? Main delegates.
            # The original test tested main's fallback. Now main doesn't have fallback, executor does.
            # So this test effectively tests start_client via main.

            with patch.dict(os.environ, {}, clear=True):
                # We need to patch the client_train in executor module, NOT main.
                # Because main imports start_client from executor.
                main(test_args)

                # Should log warning in executor
                mock_logger.warning.assert_called_with(
                    "NVFlare ClientTrain module not found. Skipping execution."
                )
