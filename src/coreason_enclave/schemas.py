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

from pydantic import BaseModel


class AggregationStrategy(str, Enum):
    """
    Strategy for aggregating model updates in the federation.
    """

    FED_AVG = "FED_AVG"  # Standard averaging
    FED_PROX = "FED_PROX"  # Handles non-IID data
    SCAFFOLD = "SCAFFOLD"  # Controls client drift


class PrivacyConfig(BaseModel):
    """
    Configuration for Differential Privacy.
    """

    mechanism: str = "DP_SGD"  # Differential Privacy
    noise_multiplier: float  # 1.0
    max_grad_norm: float  # 1.0
    target_epsilon: float  # 3.0 (Strict privacy)


class FederationJob(BaseModel):
    """
    Definition of a Federated Learning job.
    """

    job_id: UUID
    clients: List[str]  # ["node_hospital_a", "node_hospital_b"]
    min_clients: int  # 2
    rounds: int  # 50
    strategy: AggregationStrategy
    privacy: PrivacyConfig


class AttestationReport(BaseModel):
    """
    Report proving the integrity and security of the enclave.
    """

    node_id: str
    hardware_type: str  # "NVIDIA_H100_HOPPER"
    enclave_signature: str  # The hardware quote
    measurement_hash: str  # SHA256 of the running binary
    status: Literal["TRUSTED", "UNTRUSTED"]
