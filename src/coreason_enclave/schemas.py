# Copyright (c) CoReason AI. All rights reserved.
# Licensed under the Prosperity Public License 3.0.

from enum import Enum
from typing import List, Literal
from uuid import UUID

from pydantic import BaseModel, Field, model_validator


class AggregationStrategy(str, Enum):
    """
    Aggregation strategy for Federated Learning.
    """

    FED_AVG = "FED_AVG"
    FED_PROX = "FED_PROX"
    SCAFFOLD = "SCAFFOLD"


class PrivacyConfig(BaseModel):
    """
    Configuration for Differential Privacy.
    """

    mechanism: str = "DP_SGD"
    noise_multiplier: float = Field(..., gt=0.0, description="Must be positive")
    max_grad_norm: float = Field(..., gt=0.0, description="Must be positive")
    target_epsilon: float = Field(..., gt=0.0, description="Must be positive")

    @model_validator(mode="after")
    def check_epsilon_limit(self) -> "PrivacyConfig":
        """
        Validates that the privacy budget does not exceed the maximum allowed threshold.
        """
        if self.target_epsilon > 5.0:
            raise ValueError("Privacy budget (epsilon) cannot exceed 5.0")
        return self


class FederationJob(BaseModel):
    """
    Definition of a Federated Learning job.
    """

    job_id: UUID
    clients: List[str] = Field(..., min_length=1, description="List of participating client node IDs")
    min_clients: int = Field(..., ge=1, description="Minimum number of clients required")
    rounds: int = Field(..., gt=0, description="Number of training rounds")
    strategy: AggregationStrategy
    privacy: PrivacyConfig

    @model_validator(mode="after")
    def check_min_clients_consistency(self) -> "FederationJob":
        """
        Ensures that the number of provided clients meets the minimum requirement.
        """
        if len(self.clients) < self.min_clients:
            raise ValueError(
                f"Number of provided clients ({len(self.clients)}) is less than min_clients ({self.min_clients})"
            )
        return self


class AttestationReport(BaseModel):
    """
    Report from the Enclave Hardware Abstraction Layer proving code integrity.
    """

    node_id: str
    hardware_type: str
    enclave_signature: str
    measurement_hash: str
    status: Literal["TRUSTED", "UNTRUSTED"]
