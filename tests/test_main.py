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
import sys
from unittest.mock import MagicMock, patch

import pytest

from coreason_identity.models import UserContext
from coreason_enclave.main import main

# Mock sys.modules["resource"] for Windows tests if needed, similar to conftest
if sys.platform == "win32":
    try:
        import resource  # type: ignore # noqa: F401
    except ImportError:
        sys.modules["resource"] = MagicMock()


class TestMainLogic:
    """Tests for the main entry point logic (args, security, invocation)."""

    from typing import Any, Generator

    @pytest.fixture
    def mock_start_client(self) -> Generator[MagicMock, None, None]:
        """Mock the executor start_client function."""
        with patch("coreason_enclave.main.start_client") as mock:
            yield mock

    @pytest.fixture
    def mock_sys_exit(self) -> Generator[Any, None, None]:
        with patch("sys.exit") as mock:
            yield mock

    def test_main_secure_default(self, mock_start_client: MagicMock, mock_sys_exit: Any) -> None:
        """Test main runs in secure mode by default."""
        test_args = ["-w", "/tmp/ws", "-c", "conf.json"]

        with patch.dict(os.environ, {}, clear=True):
            main(test_args)

            assert os.environ["COREASON_ENCLAVE_SIMULATION"] == "false"
            # Verify Executor invoked
            mock_start_client.assert_called_once()
            call_kwargs = mock_start_client.call_args[1]
            assert call_kwargs["workspace"] == "/tmp/ws"
            assert call_kwargs["conf"] == "conf.json"
            assert isinstance(call_kwargs["context"], UserContext)

    def test_main_insecure_mode(self, mock_start_client: MagicMock, mock_sys_exit: Any) -> None:
        """Test main runs in simulation mode with flag."""
        test_args = ["-w", "/tmp/ws", "-c", "conf.json", "--insecure"]

        with patch.dict(os.environ, {}, clear=True):
            main(test_args)

            assert os.environ["COREASON_ENCLAVE_SIMULATION"] == "true"
            mock_start_client.assert_called_once()

    def test_main_simulation_mode(self, mock_start_client: MagicMock, mock_sys_exit: Any) -> None:
        """Test main runs in simulation mode with --simulation flag."""
        test_args = ["-w", "/tmp/ws", "-c", "conf.json", "--simulation"]

        with patch.dict(os.environ, {}, clear=True):
            main(test_args)

            assert os.environ["COREASON_ENCLAVE_SIMULATION"] == "true"
            mock_start_client.assert_called_once()

    def test_main_aborts_on_env_mismatch(self, mock_start_client: MagicMock, mock_sys_exit: Any) -> None:
        """Test abort if env=true but flag missing."""
        test_args = ["-w", "/tmp/ws", "-c", "conf.json"]

        with patch.dict(os.environ, {"COREASON_ENCLAVE_SIMULATION": "true"}, clear=True):
            with patch("coreason_enclave.main.logger") as mock_logger:
                main(test_args)

                # Should have called sys.exit(1)
                mock_sys_exit.assert_called_with(1)

                # Should verify critical log
                mock_logger.critical.assert_called()
                args, _ = mock_logger.critical.call_args
                assert "Security Violation" in args[0]
                assert "required '--insecure' or '--simulation' CLI flag is missing" in args[0]

    def test_insecure_flag_overrides_garbage_env(self, mock_start_client: MagicMock, mock_sys_exit: Any) -> None:
        """Test that --insecure flag overrides a garbage env var."""
        test_args = ["-w", "/tmp/ws", "-c", "conf.json", "--insecure"]

        with patch.dict(os.environ, {"COREASON_ENCLAVE_SIMULATION": "garbage_value"}, clear=True):
            main(test_args)
            assert os.environ["COREASON_ENCLAVE_SIMULATION"] == "true"
