# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_enclave

import torch
from torch import nn

from coreason_enclave.models.registry import ModelRegistry


@ModelRegistry.register("SimpleMLP")
class SimpleMLP(nn.Module):  # type: ignore[misc]
    """
    A simple Multi-Layer Perceptron for testing and basic tasks.
    """

    def __init__(self, input_dim: int = 10, hidden_dim: int = 16, output_dim: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.net(x)
