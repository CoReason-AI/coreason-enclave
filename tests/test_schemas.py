# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_enclave

import uuid

import pytest
from pydantic import ValidationError

from coreason_enclave.schemas import (
    AggregationStrategy,
    AttestationReport,
    FederationJob,
    PrivacyConfig,
)


def test_aggregation_strategy_enum() -> None:
    """Test that AggregationStrategy enum values are correct."""
    assert AggregationStrategy.FED_AVG == "FED_AVG"
    assert AggregationStrategy.FED_PROX == "FED_PROX"
    assert AggregationStrategy.SCAFFOLD == "SCAFFOLD"


def test_privacy_config_valid() -> None:
    """Test creating a valid PrivacyConfig."""
    config = PrivacyConfig(
        noise_multiplier=1.0,
        max_grad_norm=1.0,
        target_epsilon=3.0,
    )
    assert config.mechanism == "DP_SGD"
    assert config.noise_multiplier == 1.0
    assert config.max_grad_norm == 1.0
    assert config.target_epsilon == 3.0


def test_privacy_config_defaults() -> None:
    """Test PrivacyConfig defaults."""
    config = PrivacyConfig(
        noise_multiplier=0.5,
        max_grad_norm=1.5,
        target_epsilon=5.0,
    )
    assert config.mechanism == "DP_SGD"


def test_privacy_config_missing_fields() -> None:
    """Test PrivacyConfig validation for missing fields."""
    with pytest.raises(ValidationError):
        PrivacyConfig(
            # noise_multiplier missing
            max_grad_norm=1.0,
            target_epsilon=3.0,
        )  # type: ignore


def test_federation_job_valid() -> None:
    """Test creating a valid FederationJob."""
    job_id = uuid.uuid4()
    privacy = PrivacyConfig(
        noise_multiplier=1.0,
        max_grad_norm=1.0,
        target_epsilon=3.0,
    )
    job = FederationJob(
        job_id=job_id,
        clients=["node1", "node2"],
        min_clients=2,
        rounds=50,
        strategy=AggregationStrategy.FED_AVG,
        privacy=privacy,
    )
    assert job.job_id == job_id
    assert job.strategy == AggregationStrategy.FED_AVG
    assert len(job.clients) == 2


def test_federation_job_invalid_strategy() -> None:
    """Test FederationJob validation for invalid strategy."""
    job_id = uuid.uuid4()
    privacy = PrivacyConfig(
        noise_multiplier=1.0,
        max_grad_norm=1.0,
        target_epsilon=3.0,
    )
    with pytest.raises(ValidationError):
        FederationJob(
            job_id=job_id,
            clients=["node1"],
            min_clients=2,
            rounds=10,
            strategy="INVALID_STRATEGY",  # type: ignore
            privacy=privacy,
        )


def test_attestation_report_valid() -> None:
    """Test creating a valid AttestationReport."""
    report = AttestationReport(
        node_id="node1",
        hardware_type="NVIDIA_H100_HOPPER",
        enclave_signature="sig123",
        measurement_hash="hash123",
        status="TRUSTED",
    )
    assert report.status == "TRUSTED"
    assert report.node_id == "node1"


def test_attestation_report_invalid_status() -> None:
    """Test AttestationReport validation for invalid status."""
    with pytest.raises(ValidationError):
        AttestationReport(
            node_id="node1",
            hardware_type="NVIDIA_H100_HOPPER",
            enclave_signature="sig123",
            measurement_hash="hash123",
            status="SOME_OTHER_STATUS",  # type: ignore
        )
