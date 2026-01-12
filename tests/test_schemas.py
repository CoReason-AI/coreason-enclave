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

# --- PrivacyConfig Tests ---


def test_privacy_config_valid() -> None:
    config = PrivacyConfig(noise_multiplier=1.0, max_grad_norm=1.5, target_epsilon=3.0)
    assert config.noise_multiplier == 1.0
    assert config.max_grad_norm == 1.5
    assert config.target_epsilon == 3.0
    assert config.mechanism == "DP_SGD"


def test_privacy_config_negative_noise() -> None:
    with pytest.raises(ValidationError, match="noise_multiplier must be non-negative"):
        PrivacyConfig(noise_multiplier=-0.1, max_grad_norm=1.0, target_epsilon=1.0)


def test_privacy_config_zero_noise() -> None:
    """Test that zero noise is allowed (non-negative)."""
    config = PrivacyConfig(noise_multiplier=0.0, max_grad_norm=1.0, target_epsilon=1.0)
    assert config.noise_multiplier == 0.0


def test_privacy_config_small_epsilon() -> None:
    """Test very small positive epsilon is valid."""
    config = PrivacyConfig(noise_multiplier=1.0, max_grad_norm=1.0, target_epsilon=1e-9)
    assert config.target_epsilon == 1e-9


def test_privacy_config_invalid_grad_norm() -> None:
    with pytest.raises(ValidationError, match="max_grad_norm must be positive"):
        PrivacyConfig(noise_multiplier=1.0, max_grad_norm=0.0, target_epsilon=1.0)


def test_privacy_config_invalid_epsilon() -> None:
    with pytest.raises(ValidationError, match="target_epsilon must be positive"):
        PrivacyConfig(noise_multiplier=1.0, max_grad_norm=1.0, target_epsilon=0.0)


# --- FederationJob Tests ---


def test_federation_job_valid() -> None:
    job = FederationJob(
        job_id=uuid4(),
        clients=["node_a", "node_b"],
        min_clients=2,
        rounds=50,
        strategy=AggregationStrategy.FED_AVG,
        privacy=PrivacyConfig(noise_multiplier=0.5, max_grad_norm=1.0, target_epsilon=5.0),
    )
    assert len(job.clients) == 2
    assert job.rounds == 50


def test_federation_job_boundaries() -> None:
    """Test boundary conditions for rounds and clients."""
    # Rounds = 1 (Lower bound)
    job_min = FederationJob(
        job_id=uuid4(),
        clients=["node_a"],
        min_clients=1,
        rounds=1,
        strategy=AggregationStrategy.FED_AVG,
        privacy=PrivacyConfig(noise_multiplier=1.0, max_grad_norm=1.0, target_epsilon=1.0),
    )
    assert job_min.rounds == 1

    # Rounds = 10000 (Upper bound)
    job_max = FederationJob(
        job_id=uuid4(),
        clients=["node_a"],
        min_clients=1,
        rounds=10000,
        strategy=AggregationStrategy.FED_AVG,
        privacy=PrivacyConfig(noise_multiplier=1.0, max_grad_norm=1.0, target_epsilon=1.0),
    )
    assert job_max.rounds == 10000

    # Min clients = len(clients) (Upper bound)
    job_clients = FederationJob(
        job_id=uuid4(),
        clients=["node_a", "node_b"],
        min_clients=2,
        rounds=10,
        strategy=AggregationStrategy.FED_AVG,
        privacy=PrivacyConfig(noise_multiplier=1.0, max_grad_norm=1.0, target_epsilon=1.0),
    )
    assert job_clients.min_clients == 2


def test_federation_job_rounds_bounds() -> None:
    # Too low
    with pytest.raises(ValidationError, match="rounds must be between 1 and 10000"):
        FederationJob(
            job_id=uuid4(),
            clients=["node_a"],
            min_clients=1,
            rounds=0,
            strategy=AggregationStrategy.FED_AVG,
            privacy=PrivacyConfig(noise_multiplier=1.0, max_grad_norm=1.0, target_epsilon=1.0),
        )

    # Too high
    with pytest.raises(ValidationError, match="rounds must be between 1 and 10000"):
        FederationJob(
            job_id=uuid4(),
            clients=["node_a"],
            min_clients=1,
            rounds=10001,
            strategy=AggregationStrategy.FED_AVG,
            privacy=PrivacyConfig(noise_multiplier=1.0, max_grad_norm=1.0, target_epsilon=1.0),
        )


def test_federation_job_unique_clients() -> None:
    with pytest.raises(ValidationError, match="clients list must contain unique node IDs"):
        FederationJob(
            job_id=uuid4(),
            clients=["node_a", "node_a"],  # Duplicate
            min_clients=1,
            rounds=10,
            strategy=AggregationStrategy.FED_AVG,
            privacy=PrivacyConfig(noise_multiplier=1.0, max_grad_norm=1.0, target_epsilon=1.0),
        )


def test_federation_job_client_uniqueness_case_sensitivity() -> None:
    """Test that client uniqueness is case-sensitive."""
    # "node_a" and "NODE_A" are different strings, so this should be valid
    job = FederationJob(
        job_id=uuid4(),
        clients=["node_a", "NODE_A"],
        min_clients=2,
        rounds=10,
        strategy=AggregationStrategy.FED_AVG,
        privacy=PrivacyConfig(noise_multiplier=1.0, max_grad_norm=1.0, target_epsilon=1.0),
    )
    assert len(job.clients) == 2


def test_federation_job_min_clients_logic() -> None:
    # min_clients > len(clients)
    with pytest.raises(ValidationError, match="min_clients cannot be greater than the number of available clients"):
        FederationJob(
            job_id=uuid4(),
            clients=["node_a"],
            min_clients=2,
            rounds=10,
            strategy=AggregationStrategy.FED_AVG,
            privacy=PrivacyConfig(noise_multiplier=1.0, max_grad_norm=1.0, target_epsilon=1.0),
        )

    # min_clients < 1
    with pytest.raises(ValidationError, match="min_clients must be at least 1"):
        FederationJob(
            job_id=uuid4(),
            clients=["node_a"],
            min_clients=0,
            rounds=10,
            strategy=AggregationStrategy.FED_AVG,
            privacy=PrivacyConfig(noise_multiplier=1.0, max_grad_norm=1.0, target_epsilon=1.0),
        )


# --- AttestationReport Tests ---


def test_attestation_report_valid() -> None:
    valid_hash = "a" * 64
    report = AttestationReport(
        node_id="node_x",
        hardware_type="NVIDIA_H100",
        enclave_signature="sig_123",
        measurement_hash=valid_hash,
        status="TRUSTED",
    )
    assert report.measurement_hash == valid_hash


def test_attestation_report_hex_variations() -> None:
    """Test mixed case hex strings."""
    # Mixed case should be valid as hex
    mixed_hash = ("a" * 32) + ("A" * 32)
    report = AttestationReport(
        node_id="node_x",
        hardware_type="NVIDIA_H100",
        enclave_signature="sig_123",
        measurement_hash=mixed_hash,
        status="TRUSTED",
    )
    assert report.measurement_hash == mixed_hash


def test_attestation_report_invalid_hash_length() -> None:
    with pytest.raises(ValidationError, match="measurement_hash must be a 64-character hex string"):
        AttestationReport(
            node_id="node_x",
            hardware_type="NVIDIA_H100",
            enclave_signature="sig_123",
            measurement_hash="abc",  # Too short
            status="TRUSTED",
        )


def test_attestation_report_invalid_hash_chars() -> None:
    invalid_hash = "z" * 64  # 'z' is not hex
    with pytest.raises(ValidationError, match="measurement_hash must contain only hexadecimal characters"):
        AttestationReport(
            node_id="node_x",
            hardware_type="NVIDIA_H100",
            enclave_signature="sig_123",
            measurement_hash=invalid_hash,
            status="TRUSTED",
        )
