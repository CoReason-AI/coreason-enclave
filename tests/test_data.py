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

import pytest
import torch
from coreason_identity.models import UserContext

from coreason_enclave.data.loader import DataLoaderFactory
from coreason_enclave.sentry import DataSentry

valid_user_context = UserContext(
    user_id="test_user", username="tester", privacy_budget_spent=0.0, privacy_budget_limit=10.0
)


@pytest.fixture
def mock_sentry() -> MagicMock:
    sentry = MagicMock(spec=DataSentry)
    sentry.validate_input.return_value = True
    return sentry


@pytest.fixture
def loader_factory(mock_sentry: MagicMock) -> DataLoaderFactory:
    return DataLoaderFactory(mock_sentry)


def test_load_tensor_success(loader_factory: DataLoaderFactory, tmp_path: Path) -> None:
    # Create dummy tensor file
    features = torch.randn(10, 5)
    labels = torch.randn(10, 1)
    file_path = tmp_path / "data.pt"
    torch.save((features, labels), file_path)

    with patch.dict("os.environ", {"COREASON_DATA_ROOT": str(tmp_path)}):
        loader = loader_factory.get_loader("data.pt", user_context=valid_user_context, batch_size=2)

        batch = next(iter(loader))
        assert len(batch) == 2
        assert batch[0].shape == (2, 5)
        assert batch[1].shape == (2, 1)


def test_load_csv_success(loader_factory: DataLoaderFactory, tmp_path: Path) -> None:
    # Create dummy CSV
    import pandas as pd

    df = pd.DataFrame({"f1": [1.0, 2.0], "f2": [3.0, 4.0], "label": [0.0, 1.0]})
    file_path = tmp_path / "data.csv"
    df.to_csv(file_path, index=False)

    with patch.dict("os.environ", {"COREASON_DATA_ROOT": str(tmp_path)}):
        loader = loader_factory.get_loader("data.csv", user_context=valid_user_context, batch_size=1)

        batch = next(iter(loader))
        assert batch[0].shape == (1, 2)
        assert batch[1].shape == (1, 1)


def test_validation_failure(loader_factory: DataLoaderFactory) -> None:
    loader_factory.sentry.validate_input.return_value = False  # type: ignore
    with pytest.raises(ValueError, match="Input validation failed"):
        loader_factory.get_loader("bad_data.pt", user_context=valid_user_context)


def test_unsupported_format(loader_factory: DataLoaderFactory, tmp_path: Path) -> None:
    file_path = tmp_path / "data.txt"
    file_path.touch()

    with patch.dict("os.environ", {"COREASON_DATA_ROOT": str(tmp_path)}):
        with pytest.raises(ValueError, match="Unsupported file format"):
            loader_factory.get_loader("data.txt", user_context=valid_user_context)
