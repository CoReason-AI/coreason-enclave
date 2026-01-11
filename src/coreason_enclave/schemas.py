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
    noise_multiplier: float
    max_grad_norm: float
    target_epsilon: float


class FederationJob(BaseModel):
    """
    Definition of a Federated Learning Job.
    """

    job_id: UUID
    clients: List[str]
    min_clients: int
    rounds: int
    strategy: AggregationStrategy
    privacy: PrivacyConfig


class AttestationReport(BaseModel):
    """
    Report for Remote Attestation of the Enclave.
    """

    node_id: str
    hardware_type: str
    enclave_signature: str
    measurement_hash: str
    status: Literal["TRUSTED", "UNTRUSTED"]
