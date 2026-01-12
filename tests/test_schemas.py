# Copyright (c) CoReason AI. All rights reserved.
# Licensed under the Prosperity Public License 3.0.

from uuid import uuid4

import pytest
from coreason_enclave.schemas import (
    AggregationStrategy,
    AttestationReport,
    FederationJob,
    PrivacyConfig,
)
from pydantic import ValidationError


class TestPrivacyConfig:
    def test_valid_privacy_config(self) -> None:
        config = PrivacyConfig(mechanism="DP_SGD", noise_multiplier=1.0, max_grad_norm=1.0, target_epsilon=3.0)
        assert config.target_epsilon == 3.0
        assert config.noise_multiplier == 1.0

    def test_negative_noise_multiplier(self) -> None:
        with pytest.raises(ValidationError):
            PrivacyConfig(mechanism="DP_SGD", noise_multiplier=-0.1, max_grad_norm=1.0, target_epsilon=3.0)

    def test_epsilon_exceeds_limit(self) -> None:
        with pytest.raises(ValidationError) as exc:
            PrivacyConfig(mechanism="DP_SGD", noise_multiplier=1.0, max_grad_norm=1.0, target_epsilon=5.1)
        assert "Privacy budget (epsilon) cannot exceed 5.0" in str(exc.value)

    def test_zero_epsilon(self) -> None:
        with pytest.raises(ValidationError):
            PrivacyConfig(mechanism="DP_SGD", noise_multiplier=1.0, max_grad_norm=1.0, target_epsilon=0.0)

    def test_empty_mechanism(self) -> None:
        with pytest.raises(ValidationError):
            PrivacyConfig(mechanism="", noise_multiplier=1.0, max_grad_norm=1.0, target_epsilon=3.0)

    def test_epsilon_boundary(self) -> None:
        # 5.0 should pass
        config = PrivacyConfig(mechanism="DP_SGD", noise_multiplier=1.0, max_grad_norm=1.0, target_epsilon=5.0)
        assert config.target_epsilon == 5.0

        # 5.00001 should fail
        with pytest.raises(ValidationError):
            PrivacyConfig(mechanism="DP_SGD", noise_multiplier=1.0, max_grad_norm=1.0, target_epsilon=5.00001)


class TestFederationJob:
    def test_valid_federation_job(self) -> None:
        job = FederationJob(
            job_id=uuid4(),
            clients=["node_a", "node_b"],
            min_clients=2,
            rounds=10,
            strategy=AggregationStrategy.FED_AVG,
            privacy=PrivacyConfig(noise_multiplier=1.0, max_grad_norm=1.0, target_epsilon=3.0),
        )
        assert len(job.clients) == 2
        assert job.strategy == "FED_AVG"

    def test_insufficient_clients(self) -> None:
        with pytest.raises(ValidationError) as exc:
            FederationJob(
                job_id=uuid4(),
                clients=["node_a"],
                min_clients=2,
                rounds=10,
                strategy=AggregationStrategy.FED_AVG,
                privacy=PrivacyConfig(noise_multiplier=1.0, max_grad_norm=1.0, target_epsilon=3.0),
            )
        assert "Number of provided clients (1) is less than min_clients (2)" in str(exc.value)

    def test_empty_clients(self) -> None:
        with pytest.raises(ValidationError):
            FederationJob(
                job_id=uuid4(),
                clients=[],
                min_clients=1,
                rounds=10,
                strategy=AggregationStrategy.FED_AVG,
                privacy=PrivacyConfig(noise_multiplier=1.0, max_grad_norm=1.0, target_epsilon=3.0),
            )

    def test_invalid_strategy(self) -> None:
        with pytest.raises(ValidationError):
            FederationJob(
                job_id=uuid4(),
                clients=["node_a"],
                min_clients=1,
                rounds=10,
                strategy="INVALID_STRATEGY",  # type: ignore
                privacy=PrivacyConfig(noise_multiplier=1.0, max_grad_norm=1.0, target_epsilon=3.0),
            )

    def test_duplicate_clients(self) -> None:
        with pytest.raises(ValidationError) as exc:
            FederationJob(
                job_id=uuid4(),
                clients=["node_a", "node_a"],
                min_clients=1,
                rounds=10,
                strategy=AggregationStrategy.FED_AVG,
                privacy=PrivacyConfig(noise_multiplier=1.0, max_grad_norm=1.0, target_epsilon=3.0),
            )
        assert "Client IDs must be unique" in str(exc.value)

    def test_max_rounds_exceeded(self) -> None:
        with pytest.raises(ValidationError):
            FederationJob(
                job_id=uuid4(),
                clients=["node_a"],
                min_clients=1,
                rounds=10001,
                strategy=AggregationStrategy.FED_AVG,
                privacy=PrivacyConfig(noise_multiplier=1.0, max_grad_norm=1.0, target_epsilon=3.0),
            )


class TestAttestationReport:
    def test_valid_attestation_report(self) -> None:
        valid_hash = "a" * 64
        report = AttestationReport(
            node_id="node_1",
            hardware_type="NVIDIA_H100_HOPPER",
            enclave_signature="sig_123",
            measurement_hash=valid_hash,
            status="TRUSTED",
        )
        assert report.status == "TRUSTED"
        assert report.node_id == "node_1"

    def test_invalid_status(self) -> None:
        valid_hash = "a" * 64
        with pytest.raises(ValidationError):
            AttestationReport(
                node_id="node_1",
                hardware_type="NVIDIA_H100_HOPPER",
                enclave_signature="sig_123",
                measurement_hash=valid_hash,
                status="MAYBE_TRUSTED",  # type: ignore
            )

    def test_empty_string_fields(self) -> None:
        valid_hash = "a" * 64
        with pytest.raises(ValidationError):
            AttestationReport(
                node_id="",
                hardware_type="NVIDIA_H100_HOPPER",
                enclave_signature="sig_123",
                measurement_hash=valid_hash,
                status="TRUSTED",
            )

    def test_invalid_hash_format(self) -> None:
        # Too short
        with pytest.raises(ValidationError):
            AttestationReport(
                node_id="node_1",
                hardware_type="NVIDIA_H100_HOPPER",
                enclave_signature="sig_123",
                measurement_hash="abc",
                status="TRUSTED",
            )

        # Not hex
        with pytest.raises(ValidationError):
            AttestationReport(
                node_id="node_1",
                hardware_type="NVIDIA_H100_HOPPER",
                enclave_signature="sig_123",
                measurement_hash="z" * 64,
                status="TRUSTED",
            )
