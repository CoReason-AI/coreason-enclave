# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_enclave

import pytest
import torch
from torch import nn

from coreason_enclave.models.registry import ModelRegistry
from coreason_enclave.models.simple_mlp import SimpleMLP


def test_registry_registration() -> None:
    # Verify SimpleMLP is auto-registered via import
    model_cls = ModelRegistry.get("SimpleMLP")
    assert model_cls == SimpleMLP
    assert issubclass(model_cls, nn.Module)


def test_registry_manual_registration() -> None:
    @ModelRegistry.register("TestModel")
    class TestModel(nn.Module):  # type: ignore[misc]
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x

    assert ModelRegistry.get("TestModel") == TestModel


def test_registry_overwrite_warning(caplog: pytest.LogCaptureFixture) -> None:
    from coreason_enclave.utils.logger import logger

    # Propagate logs to caplog
    logger.add(caplog.handler, format="{message}", level="WARNING")

    @ModelRegistry.register("DuplicateModel")
    class ModelA(nn.Module):  # type: ignore[misc]
        pass

    @ModelRegistry.register("DuplicateModel")
    class ModelB(nn.Module):  # type: ignore[misc]
        pass

    assert ModelRegistry.get("DuplicateModel") == ModelB
    # Loguru uses a different mechanism, but caplog should catch if propagated or configured correctly.
    # Alternatively, verify side effect or mock logger.
    # Since we added caplog.handler, it should work.
    assert "already registered. Overwriting" in caplog.text


def test_registry_not_found() -> None:
    import pytest

    with pytest.raises(ValueError, match="Model 'Unknown' not found"):
        ModelRegistry.get("Unknown")


def test_simple_mlp_structure() -> None:
    import torch

    model = SimpleMLP(input_dim=5, hidden_dim=10, output_dim=2)
    dummy_input = torch.randn(1, 5)
    output = model(dummy_input)
    assert output.shape == (1, 2)
