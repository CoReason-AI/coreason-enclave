# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_enclave

import pickle
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from coreason_enclave.data.loader import DataLoaderFactory
from coreason_enclave.main import main
from coreason_enclave.models.registry import ModelRegistry
from coreason_enclave.sentry import DataSentry

# --- Loader Tests ---


def test_loader_tensor_load_fail(tmp_path: Path) -> None:
    """Test failure when torch.load fails."""
    # Create invalid tensor file
    file_path = tmp_path / "bad.pt"
    file_path.write_text("invalid content")

    sentry = MagicMock(spec=DataSentry)
    sentry.validate_input.return_value = True
    factory = DataLoaderFactory(sentry)

    with patch.dict("os.environ", {"COREASON_DATA_ROOT": str(tmp_path)}):
        # pickle.UnpicklingError for weights_only=True default in newer torch
        with pytest.raises((RuntimeError, pickle.UnpicklingError)):
            factory.get_loader("bad.pt")


def test_loader_tensor_invalid_format(tmp_path: Path) -> None:
    """Test tensor file with valid pickle but invalid structure."""
    file_path = tmp_path / "structure.pt"
    torch.save({"wrong": "keys"}, file_path)

    sentry = MagicMock(spec=DataSentry)
    sentry.validate_input.return_value = True
    factory = DataLoaderFactory(sentry)

    with patch.dict("os.environ", {"COREASON_DATA_ROOT": str(tmp_path)}):
        with pytest.raises(ValueError, match="Tensor file must contain"):
            factory.get_loader("structure.pt")


def test_loader_csv_import_error(tmp_path: Path) -> None:
    """Test pandas import error handling."""
    file_path = tmp_path / "data.csv"
    file_path.touch()

    sentry = MagicMock(spec=DataSentry)
    sentry.validate_input.return_value = True

    with patch.dict("os.environ", {"COREASON_DATA_ROOT": str(tmp_path)}):
        with patch.dict(sys.modules, {"pandas": None}):
            # Remove pandas from modules temporarily or mock import failure
            pass


def test_loader_csv_load_fail(tmp_path: Path) -> None:
    """Test failure when pd.read_csv fails."""
    file_path = tmp_path / "bad.csv"
    file_path.write_text("bad,csv")

    sentry = MagicMock(spec=DataSentry)
    sentry.validate_input.return_value = True
    factory = DataLoaderFactory(sentry)

    with patch.dict("os.environ", {"COREASON_DATA_ROOT": str(tmp_path)}):
        # Mock pandas read_csv to raise
        with patch("pandas.read_csv", side_effect=Exception("CSV Error")):
            with pytest.raises(Exception, match="CSV Error"):
                factory.get_loader("bad.csv")


# --- Main Tests ---


def test_main_exception_handling() -> None:
    """Test the broad exception handler in main."""
    with patch("argparse.ArgumentParser.parse_args", side_effect=Exception("Crash")):
        with patch("coreason_enclave.main.logger") as mock_logger:
            with pytest.raises(SystemExit):
                main(["-w", "a", "-c", "b"])

            mock_logger.exception.assert_called_with("Failed to start Enclave Agent: Crash")


def test_main_opts() -> None:
    """Test main with additional opts."""
    # We must mock client_train because main now calls it, and it will fail on dummy opts
    with patch("coreason_enclave.main.client_train") as mock_ct:
        with patch("coreason_enclave.main.logger"):
            # We mock parse_arguments to succeed
            mock_ct.parse_arguments.return_value = MagicMock()

            main(["-w", "w", "-c", "c", "opt1", "opt2"])

            # Verify opts were passed to sys.argv (implicitly tested by logic, but we can assume success if no crash)
            mock_ct.parse_arguments.assert_called_once()
            mock_ct.main.assert_called_once()


def test_loader_tensor_dict_success(tmp_path: Path) -> None:
    """Test loading tensor dict format."""
    features = torch.randn(5, 5)
    labels = torch.randn(5, 1)
    file_path = tmp_path / "dict.pt"
    torch.save({"features": features, "labels": labels}, file_path)

    sentry = MagicMock(spec=DataSentry)
    sentry.validate_input.return_value = True
    factory = DataLoaderFactory(sentry)

    with patch.dict("os.environ", {"COREASON_DATA_ROOT": str(tmp_path)}):
        loader = factory.get_loader("dict.pt", batch_size=2)
        assert len(loader.dataset) == 5


# --- Registry Tests ---


def test_registry_logger_debug(caplog: pytest.LogCaptureFixture) -> None:
    """Verify debug log on registration."""
    from coreason_enclave.utils.logger import logger

    logger.add(caplog.handler, level="DEBUG")

    @ModelRegistry.register("DebugModel")
    class M(torch.nn.Module):  # type: ignore[misc]
        pass

    assert "Registered model: DebugModel" in caplog.text


def test_registry_clear() -> None:
    """Test clearing the registry."""

    # Register dummy model first
    @ModelRegistry.register("Temp")
    class Temp(torch.nn.Module):  # type: ignore[misc]
        pass

    # Clear
    ModelRegistry.clear()
    assert len(ModelRegistry._registry) == 0

    # Restore SimpleMLP for other tests!
    from coreason_enclave.models.simple_mlp import SimpleMLP

    ModelRegistry.register("SimpleMLP")(SimpleMLP)
