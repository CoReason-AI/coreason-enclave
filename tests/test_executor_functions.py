from unittest.mock import MagicMock, patch

import pytest
from coreason_identity.models import UserContext

from coreason_enclave.federation.executor import run_federated_training, start_client, start_server


class TestExecutorFunctions:
    @pytest.fixture
    def context(self) -> UserContext:
        return UserContext(
            sub="test-user",
            email="test@coreason.ai",
            permissions=[],
            project_context="test",
        )

    def test_start_client_success(self, context: UserContext) -> None:
        """Test start_client success path."""
        with patch("coreason_enclave.federation.executor.client_train") as mock_client_train:
            # Mock parse_arguments to return args
            mock_client_train.parse_arguments.return_value = MagicMock()

            start_client(context, "/tmp/ws", "config.json", ["opt1"])

            mock_client_train.parse_arguments.assert_called_once()
            mock_client_train.main.assert_called_once()

    def test_start_client_no_context(self) -> None:
        """Test start_client raises ValueError without context."""
        with pytest.raises(ValueError, match="UserContext is required"):
            start_client(None, "/tmp/ws", "config.json")

    def test_start_client_no_module(self, context: UserContext) -> None:
        """Test start_client logs warning if client_train missing."""
        with patch("coreason_enclave.federation.executor.client_train", None):
            with patch("coreason_enclave.federation.executor.logger") as mock_logger:
                start_client(context, "/tmp/ws", "config.json")
                mock_logger.warning.assert_called_with("NVFlare ClientTrain module not found. Skipping execution.")

    def test_start_server_success(self, context: UserContext) -> None:
        """Test start_server success path (stub)."""
        # Just verify it doesn't crash and logs
        with patch("coreason_enclave.federation.executor.logger") as mock_logger:
            start_server(context)
            mock_logger.info.assert_any_call(
                "Initializing Federation Executor",
                user_id=context.sub,
                action="start_server",
            )

    def test_start_server_no_context(self) -> None:
        """Test start_server raises ValueError without context."""
        with pytest.raises(ValueError, match="UserContext is required"):
            start_server(None)

    def test_run_federated_training_success(self, context: UserContext) -> None:
        """Test run_federated_training success path (stub)."""
        with patch("coreason_enclave.federation.executor.logger") as mock_logger:
            run_federated_training(context)
            mock_logger.info.assert_any_call(
                "Initializing Federation Executor",
                user_id=context.sub,
                action="run_federated_training",
            )

    def test_run_federated_training_no_context(self) -> None:
        """Test run_federated_training raises ValueError without context."""
        with pytest.raises(ValueError, match="UserContext is required"):
            run_federated_training(None)
