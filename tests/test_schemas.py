# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_enclave

from uuid import uuid4

import pytest
from pydantic import ValidationError

from coreason_enclave.schemas import (
    AggregationStrategy,
    AttestationReport,
    FederationJob,
    PrivacyConfig,
)


def test_aggregation_strategy_enum() -> None:
    """Test AggregationStrategy Enum values."""
    assert AggregationStrategy.FED_AVG == "FED_AVG"
    assert AggregationStrategy.FED_PROX == "FED_PROX"
    assert AggregationStrategy.SCAFFOLD == "SCAFFOLD"


def test_privacy_config_valid() -> None:
    """Test PrivacyConfig with valid data."""
    config = PrivacyConfig(
        mechanism="DP_SGD",
        noise_multiplier=1.0,
        max_grad_norm=1.0,
        target_epsilon=3.0,
    )
    assert config.mechanism == "DP_SGD"
    assert config.noise_multiplier == 1.0
    assert config.max_grad_norm == 1.0
    assert config.target_epsilon == 3.0


def test_privacy_config_defaults() -> None:
    """Test PrivacyConfig default values."""
    config = PrivacyConfig(
        noise_multiplier=0.5,
        max_grad_norm=1.5,
        target_epsilon=5.0,
    )
    assert config.mechanism == "DP_SGD"  # Check default
    assert config.noise_multiplier == 0.5


def test_privacy_config_invalid_types() -> None:
    """Test PrivacyConfig with invalid types."""
    with pytest.raises(ValidationError):
        PrivacyConfig(
            noise_multiplier="invalid",  # type: ignore
            max_grad_norm=1.0,
            target_epsilon=3.0,
        )


def test_federation_job_valid() -> None:
    """Test FederationJob with valid data."""
    job_id = uuid4()
    privacy = PrivacyConfig(
        noise_multiplier=1.0,
        max_grad_norm=1.0,
        target_epsilon=3.0,
    )
    job = FederationJob(
        job_id=job_id,
        clients=["hospital_a", "hospital_b"],
        min_clients=2,
        rounds=50,
        strategy=AggregationStrategy.FED_AVG,
        privacy=privacy,
    )
    assert job.job_id == job_id
    assert job.clients == ["hospital_a", "hospital_b"]
    assert job.min_clients == 2
    assert job.rounds == 50
    assert job.strategy == AggregationStrategy.FED_AVG
    assert job.privacy == privacy


def test_federation_job_invalid_enum() -> None:
    """Test FederationJob with invalid aggregation strategy."""
    job_id = uuid4()
    privacy = PrivacyConfig(
        noise_multiplier=1.0,
        max_grad_norm=1.0,
        target_epsilon=3.0,
    )
    with pytest.raises(ValidationError):
        FederationJob(
            job_id=job_id,
            clients=["hospital_a"],
            min_clients=1,
            rounds=10,
            strategy="INVALID_STRATEGY",  # type: ignore
            privacy=privacy,
        )


def test_attestation_report_valid() -> None:
    """Test AttestationReport with valid data."""
    report = AttestationReport(
        node_id="node_123",
        hardware_type="NVIDIA_H100_HOPPER",
        enclave_signature="sig_123",
        measurement_hash="hash_123",
        status="TRUSTED",
    )
    assert report.node_id == "node_123"
    assert report.status == "TRUSTED"


def test_attestation_report_invalid_status() -> None:
    """Test AttestationReport with invalid status literal."""
    with pytest.raises(ValidationError):
        AttestationReport(
            node_id="node_123",
            hardware_type="NVIDIA_H123_HOPPER",
            enclave_signature="sig_123",
            measurement_hash="hash_123",
            status="MAYBE",  # type: ignore
        )
