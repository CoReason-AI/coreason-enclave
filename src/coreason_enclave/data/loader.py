# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_enclave

"""
Data Loading Utilities.

Provides a factory for securely loading data from disk into PyTorch DataLoaders.
Integrates with DataSentry to ensure path security.
"""

import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from coreason_identity.models import UserContext
from coreason_identity.exceptions import IdentityVerificationError
from coreason_enclave.sentry import DataSentry
from coreason_enclave.utils.logger import logger


class DataLoaderFactory:
    """
    Factory to create PyTorch DataLoaders from secured dataset IDs.

    Integrates with DataSentry for path validation to prevent traversal attacks.
    """

    def __init__(self, sentry: DataSentry):
        """
        Initialize the DataLoaderFactory.

        Args:
            sentry (DataSentry): The DataSentry instance for validation.
        """
        self.sentry = sentry

    def get_loader(
        self, dataset_id: str, user_context: UserContext, batch_size: int = 32
    ) -> DataLoader:
        """
        Load data and return a PyTorch DataLoader.

        Args:
            dataset_id (str): The ID of the dataset (e.g. "data.csv").
            user_context (UserContext): The identity of the job owner.
            batch_size (int): Batch size for training.

        Returns:
            DataLoader: The PyTorch DataLoader.

        Raises:
            FileNotFoundError: If data is missing.
            ValueError: If data is invalid or format is unsupported.
            IdentityVerificationError: If user is not authorized.
        """
        # 0. Authorize Access
        # Simple check: UserContext must be valid (already validated by Pydantic)
        # In a real scenario, we might check permissions against the dataset_id here.
        if not user_context.user_id:
            logger.error("Access denied: Invalid user context")
            raise IdentityVerificationError("Invalid user context: missing user_id")

        logger.info(f"Authorizing data access for user: {user_context.user_id}")

        # 1. Validate Path Security & Existence
        # We assume schema is None for simple file existence check,
        # or we could enforce a schema if we had one.
        if not self.sentry.validate_input(dataset_id, schema=None):
            raise ValueError(f"Input validation failed for {dataset_id}")

        # 2. Resolve Path
        data_root = os.environ.get("COREASON_DATA_ROOT", ".")
        file_path = (Path(data_root) / dataset_id).resolve()

        # 3. Load Data based on extension
        if file_path.suffix == ".pt":
            return self._load_tensor(file_path, batch_size)
        elif file_path.suffix == ".csv":
            return self._load_csv(file_path, batch_size)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

    def _load_tensor(self, path: Path, batch_size: int) -> DataLoader:
        """Load a .pt file containing (features, labels)."""
        logger.info(f"Loading Tensor dataset from {path}")
        try:
            data = torch.load(path)
            if isinstance(data, (tuple, list)) and len(data) == 2:
                features, labels = data
            elif isinstance(data, dict) and "features" in data and "labels" in data:
                features, labels = data["features"], data["labels"]
            else:
                raise ValueError(
                    "Tensor file must contain tuple (features, labels) or dict with keys 'features', 'labels'"
                )

            dataset = TensorDataset(features, labels)
            return DataLoader(dataset, batch_size=batch_size, shuffle=True)
        except Exception as e:
            logger.error(f"Failed to load tensor data: {e}")
            raise

    def _load_csv(self, path: Path, batch_size: int) -> DataLoader:
        """
        Load a CSV file.

        Assumes last column is label, rest are features.
        Values must be numeric.
        """
        logger.info(f"Loading CSV dataset from {path}")
        try:
            import pandas as pd

            df = pd.read_csv(path)

            # Simple assumption: numeric data only
            features = torch.tensor(df.iloc[:, :-1].values, dtype=torch.float32)
            labels = torch.tensor(df.iloc[:, -1].values, dtype=torch.float32).unsqueeze(1)  # shape (N, 1)

            dataset = TensorDataset(features, labels)
            return DataLoader(dataset, batch_size=batch_size, shuffle=True)
        except ImportError as e:  # pragma: no cover
            logger.error("Pandas is required for CSV loading")
            raise ImportError("Pandas is required for CSV loading") from e
        except Exception as e:
            logger.error(f"Failed to load CSV data: {e}")
            raise
