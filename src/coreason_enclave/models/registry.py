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
Model Registry.

Allows dynamic, safe instantiation of torch.nn.Module classes using string identifiers.
This prevents arbitrary code execution vulnerabilities associated with pickling or `eval()`.
"""

from typing import Callable, Type

from torch import nn

from coreason_enclave.utils.logger import logger


class ModelRegistry:
    """
    Registry for machine learning models.

    Allows dynamic instantiation of models by string ID (safe code loading).
    """

    _registry: dict[str, Type[nn.Module]] = {}

    @classmethod
    def register(cls, name: str) -> "Callable[[Type[nn.Module]], Type[nn.Module]]":
        """
        Decorator to register a model class.

        Args:
            name (str): The ID to register the model under.
        """

        def decorator(model_cls: Type[nn.Module]) -> Type[nn.Module]:
            if name in cls._registry:
                logger.warning(f"Model '{name}' already registered. Overwriting.")
            cls._registry[name] = model_cls
            logger.debug(f"Registered model: {name} -> {model_cls.__name__}")
            return model_cls

        return decorator

    @classmethod
    def get(cls, name: str) -> Type[nn.Module]:
        """
        Get a model class by name.

        Args:
            name (str): The model ID.

        Returns:
            Type[nn.Module]: The model class.

        Raises:
            ValueError: If the model is not found.
        """
        if name not in cls._registry:
            raise ValueError(f"Model '{name}' not found in registry. Available: {list(cls._registry.keys())}")
        return cls._registry[name]

    @classmethod
    def clear(cls) -> None:
        """Clear the registry (useful for testing)."""
        cls._registry.clear()
