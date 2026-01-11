# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_enclave

from enum import Enum
from typing import List, Literal
from uuid import UUID

from pydantic import BaseModel, Field, model_validator


class AggregationStrategy(str, Enum):
    """
    Aggregation Strategy for Federated Learning.
    """

    FED_AVG = "FED_AVG"  # Standard averaging
    FED_PROX = "FED_PROX"  # Handles non-IID data
    SCAFFOLD = "SCAFFOLD"  # Controls client drift


class PrivacyConfig(BaseModel):
    """
    Configuration for Differential Privacy.
    """

    mechanism: str = "DP_SGD"
    noise_multiplier: float = Field(..., ge=0.0)
    max_grad_norm: float = Field(..., gt=0.0)
    target_epsilon: float = Field(..., gt=0.0)


class FederationJob(BaseModel):
    """
    Definition of a Federated Learning Job.
    """

    job_id: UUID
    clients: List[str] = Field(..., min_length=1)
    min_clients: int = Field(..., ge=1)
    rounds: int = Field(..., ge=1)
    strategy: AggregationStrategy
    privacy: PrivacyConfig

    @model_validator(mode="after")
    def check_min_clients_satisfied(self) -> "FederationJob":
        """
        Validate that enough clients are provided to satisfy min_clients.
        """
        if len(self.clients) < self.min_clients:
            raise ValueError(f"Number of clients ({len(self.clients)}) is less than min_clients ({self.min_clients})")
        return self


class AttestationReport(BaseModel):
    """
    Report for Remote Attestation of the Enclave.
    """

    node_id: str = Field(..., min_length=1)
    hardware_type: str = Field(..., min_length=1)
    enclave_signature: str = Field(..., min_length=1)
    measurement_hash: str = Field(..., min_length=1)
    status: Literal["TRUSTED", "UNTRUSTED"]
