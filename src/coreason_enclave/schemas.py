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
Data schemas for the Coreason Enclave.

Defines the structures for federation jobs, privacy configuration, and attestation reports.
"""

from enum import Enum
from typing import List, Literal
from uuid import UUID

from coreason_identity.models import UserContext
from pydantic import BaseModel, Field, field_validator, model_validator


class AggregationStrategy(str, Enum):
    """
    Strategy for aggregating model updates in Federated Learning.

    Attributes:
        FED_AVG: Standard Federated Averaging.
        FED_PROX: FedProx strategy for robustness to non-IID data (proximal term).
        SCAFFOLD: SCAFFOLD strategy for controlling client drift.
    """

    FED_AVG = "FED_AVG"
    FED_PROX = "FED_PROX"
    SCAFFOLD = "SCAFFOLD"


class PrivacyConfig(BaseModel):
    """
    Configuration for Differential Privacy (DP-SGD).

    Attributes:
        mechanism (str): Privacy mechanism to use (default: "DP_SGD").
        noise_multiplier (float): Amount of noise to add (sigma).
        max_grad_norm (float): Maximum gradient norm for clipping (C).
        target_epsilon (float): Target privacy budget (epsilon).
    """

    mechanism: str = Field(default="DP_SGD", description="Privacy mechanism to use")
    noise_multiplier: float = Field(..., description="Amount of noise to add")
    max_grad_norm: float = Field(..., description="Maximum gradient norm for clipping")
    target_epsilon: float = Field(..., description="Target privacy budget (epsilon)")

    @field_validator("noise_multiplier")
    @classmethod
    def validate_noise_multiplier(cls, v: float) -> float:
        """Validate that noise_multiplier is non-negative."""
        if v < 0:
            raise ValueError("noise_multiplier must be non-negative")
        return v

    @field_validator("max_grad_norm")
    @classmethod
    def validate_max_grad_norm(cls, v: float) -> float:
        """Validate that max_grad_norm is positive."""
        if v <= 0:
            raise ValueError("max_grad_norm must be positive")
        return v

    @field_validator("target_epsilon")
    @classmethod
    def validate_target_epsilon(cls, v: float) -> float:
        """Validate that target_epsilon is positive."""
        if v <= 0:
            raise ValueError("target_epsilon must be positive")
        return v


class FederationJob(BaseModel):
    """
    Definition of a Federated Learning job.

    Attributes:
        job_id (UUID): Unique identifier for the job.
        clients (List[str]): List of participating client node IDs.
        min_clients (int): Minimum number of clients required to proceed.
        rounds (int): Number of training rounds.
        dataset_id (str): Identifier/path for the training dataset (relative to data root).
        model_arch (str): ID of the model architecture in the Registry.
        strategy (AggregationStrategy): Aggregation strategy (FED_AVG, FED_PROX, SCAFFOLD).
        proximal_mu (float): Proximal term coefficient for FedProx (default: 0.01).
        privacy (PrivacyConfig): Differential Privacy configuration.
    """

    job_id: UUID
    clients: List[str] = Field(..., description="List of participating client node IDs")
    min_clients: int = Field(..., description="Minimum number of clients required")
    rounds: int = Field(..., description="Number of training rounds")
    dataset_id: str = Field(..., description="Identifier/path for the training dataset")
    model_arch: str = Field(..., description="ID of the model architecture in the Registry")
    strategy: AggregationStrategy
    proximal_mu: float = Field(0.01, description="Proximal term coefficient for FedProx")
    privacy: PrivacyConfig
    user_context: UserContext = Field(..., description="Identity of the job owner")

    @field_validator("proximal_mu")
    @classmethod
    def validate_proximal_mu(cls, v: float) -> float:
        """Validate that proximal_mu is non-negative."""
        if v < 0:
            raise ValueError("proximal_mu must be non-negative")
        return v

    @field_validator("dataset_id")
    @classmethod
    def validate_dataset_id(cls, v: str) -> str:
        """Validate that dataset_id is not empty."""
        if not v or not v.strip():
            raise ValueError("dataset_id cannot be empty")
        return v

    @field_validator("model_arch")
    @classmethod
    def validate_model_arch(cls, v: str) -> str:
        """Validate that model_arch is not empty."""
        if not v or not v.strip():
            raise ValueError("model_arch cannot be empty")
        return v

    @field_validator("rounds")
    @classmethod
    def validate_rounds(cls, v: int) -> int:
        """Validate that rounds is between 1 and 10000."""
        if not (1 <= v <= 10000):
            raise ValueError("rounds must be between 1 and 10000")
        return v

    @field_validator("clients")
    @classmethod
    def validate_clients_unique(cls, v: List[str]) -> List[str]:
        """Validate that client IDs are unique."""
        if len(v) != len(set(v)):
            raise ValueError("clients list must contain unique node IDs")
        return v

    @model_validator(mode="after")
    def validate_min_clients_logic(self) -> "FederationJob":
        """Validate consistency between min_clients and clients list."""
        if self.min_clients < 1:
            raise ValueError("min_clients must be at least 1")
        if self.min_clients > len(self.clients):
            raise ValueError("min_clients cannot be greater than the number of available clients")
        return self


class AttestationReport(BaseModel):
    """
    Report from the Trusted Execution Environment (TEE).

    Contains the cryptographic evidence required for Remote Attestation.

    Attributes:
        node_id (str): Identifier of the node.
        hardware_type (str): Type of hardware (e.g., NVIDIA_H100_HOPPER).
        enclave_signature (str): The hardware quote/signature.
        measurement_hash (str): SHA256 hash of the running binary (measurement).
        status (Literal["TRUSTED", "UNTRUSTED"]): Trust status.
    """

    node_id: str
    hardware_type: str = Field(..., description="e.g. NVIDIA_H100_HOPPER")
    enclave_signature: str = Field(..., description="The hardware quote")
    measurement_hash: str = Field(..., description="SHA256 of the running binary")
    status: Literal["TRUSTED", "UNTRUSTED"]

    @field_validator("measurement_hash")
    @classmethod
    def validate_hash_format(cls, v: str) -> str:
        """Validate that measurement_hash is a valid SHA256 hex string."""
        # Check length for SHA256 hex string (64 characters)
        if len(v) != 64:
            raise ValueError("measurement_hash must be a 64-character hex string (SHA256)")
        # Check if it contains only hex characters
        try:
            int(v, 16)
        except ValueError as e:
            raise ValueError("measurement_hash must contain only hexadecimal characters") from e
        return v
