# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_enclave

import json
import uuid
from typing import Any, Dict, Generator
from unittest.mock import MagicMock, patch

import pytest
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import ReturnCode, Shareable
from nvflare.apis.signal import Signal

from coreason_enclave.federation.executor import CoreasonExecutor
from coreason_enclave.schemas import AttestationReport


class TestCoreasonExecutorSecurity:
    @pytest.fixture
    def mock_job_config(self) -> Dict[str, Any]:
        return {
            "job_id": str(uuid.uuid4()),
            "clients": ["client1", "client2"],
            "min_clients": 2,
            "rounds": 10,
            "strategy": "FED_AVG",
            "privacy": {"mechanism": "DP_SGD", "noise_multiplier": 1.0, "max_grad_norm": 1.0, "target_epsilon": 3.0},
        }

    @pytest.fixture
    def mock_attestation_provider(self) -> Generator[MagicMock, None, None]:
        with patch("coreason_enclave.federation.executor.get_attestation_provider") as mock_get:
            provider = MagicMock()
            mock_get.return_value = provider
            # Default to TRUSTED
            provider.attest.return_value = AttestationReport(
                node_id="test_node",
                hardware_type="SIMULATION",
                enclave_signature="sig",
                measurement_hash="0" * 64,
                status="TRUSTED",
            )
            yield provider

    @pytest.fixture
    def executor(self, mock_attestation_provider: MagicMock) -> CoreasonExecutor:
        # We need to ensure the mock is active when __init__ is called
        return CoreasonExecutor(training_task_name="train_task")

    @pytest.fixture
    def mock_fl_ctx(self) -> MagicMock:
        return MagicMock(spec=FLContext)

    @pytest.fixture
    def mock_signal(self) -> MagicMock:
        signal = MagicMock(spec=Signal)
        signal.triggered = False
        return signal

    @pytest.fixture
    def valid_shareable(self, mock_job_config: Dict[str, Any]) -> Shareable:
        s = Shareable()
        s.set_header("job_config", json.dumps(mock_job_config))
        return s

    def test_secure_execution_success(
        self,
        executor: CoreasonExecutor,
        valid_shareable: Shareable,
        mock_fl_ctx: MagicMock,
        mock_signal: MagicMock,
    ) -> None:
        """Test happy path: trusted hardware, valid config, valid data."""

        # Mock DataSentry to avoid filesystem checks
        executor.sentry = MagicMock()
        executor.sentry.validate_input.return_value = True
        executor.sentry.sanitize_output.return_value = {"params": {"a": 1}}

        result = executor.execute("train_task", valid_shareable, mock_fl_ctx, mock_signal)

        assert result.get_return_code() == ReturnCode.OK
        # Verify flow
        executor.attestation_provider.attest.assert_called_once()  # type: ignore
        executor.sentry.validate_input.assert_called_once()
        executor.sentry.sanitize_output.assert_called_once()

    def test_secure_execution_untrusted_hardware(
        self,
        executor: CoreasonExecutor,
        valid_shareable: Shareable,
        mock_fl_ctx: MagicMock,
        mock_signal: MagicMock,
    ) -> None:
        """Test failure when hardware is untrusted."""
        executor.attestation_provider.attest.return_value = AttestationReport(  # type: ignore
            node_id="test_node",
            hardware_type="SIMULATION",
            enclave_signature="sig",
            measurement_hash="0" * 64,
            status="UNTRUSTED",
        )

        result = executor.execute("train_task", valid_shareable, mock_fl_ctx, mock_signal)

        assert result.get_return_code() == ReturnCode.EXECUTION_EXCEPTION
        # Verify we stopped early
        # Since implementation details might vary, we check the exception was caught

    def test_secure_execution_missing_config(
        self,
        executor: CoreasonExecutor,
        mock_fl_ctx: MagicMock,
        mock_signal: MagicMock,
    ) -> None:
        """Test failure when job_config is missing."""
        executor.attestation_provider.attest.return_value.status = "TRUSTED"  # type: ignore
        shareable = Shareable()  # No header

        result = executor.execute("train_task", shareable, mock_fl_ctx, mock_signal)
        assert result.get_return_code() == ReturnCode.BAD_TASK_DATA

    def test_secure_execution_invalid_config(
        self,
        executor: CoreasonExecutor,
        mock_fl_ctx: MagicMock,
        mock_signal: MagicMock,
    ) -> None:
        """Test failure when job_config is invalid."""
        executor.attestation_provider.attest.return_value.status = "TRUSTED"  # type: ignore
        shareable = Shareable()
        shareable.set_header("job_config", "INVALID_JSON")

        result = executor.execute("train_task", shareable, mock_fl_ctx, mock_signal)
        assert result.get_return_code() == ReturnCode.BAD_TASK_DATA

    def test_secure_execution_input_validation_failure(
        self,
        executor: CoreasonExecutor,
        valid_shareable: Shareable,
        mock_fl_ctx: MagicMock,
        mock_signal: MagicMock,
    ) -> None:
        """Test failure when input validation fails."""
        executor.attestation_provider.attest.return_value.status = "TRUSTED"  # type: ignore

        # Mock Sentry to raise exception
        executor.sentry = MagicMock()
        executor.sentry.validate_input.side_effect = ValueError("Invalid data")

        result = executor.execute("train_task", valid_shareable, mock_fl_ctx, mock_signal)
        assert result.get_return_code() == ReturnCode.EXECUTION_EXCEPTION

    def test_secure_execution_sanitation_failure(
        self,
        executor: CoreasonExecutor,
        valid_shareable: Shareable,
        mock_fl_ctx: MagicMock,
        mock_signal: MagicMock,
    ) -> None:
        """Test failure when output sanitation fails."""
        executor.attestation_provider.attest.return_value.status = "TRUSTED"  # type: ignore

        # Mock Sentry
        executor.sentry = MagicMock()
        executor.sentry.validate_input.return_value = True
        executor.sentry.sanitize_output.side_effect = RuntimeError("Leakage detected")

        result = executor.execute("train_task", valid_shareable, mock_fl_ctx, mock_signal)
        assert result.get_return_code() == ReturnCode.EXECUTION_EXCEPTION
