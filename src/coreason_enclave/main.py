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
Entry point for the Coreason Enclave Agent.

This module bootstraps the NVFlare client, enforces security policies (hardware attestation vs. simulation),
and acts as the "Secure Compute Wrapper" for the training process.
"""

import argparse
import os
import sys
import threading
from typing import Optional

import uvicorn

# Workaround for NVFlare Windows issue
if sys.platform == "win32":  # pragma: no cover
    from unittest.mock import MagicMock

    try:
        import resource  # noqa: F401
    except ImportError:
        sys.modules["resource"] = MagicMock()

from coreason_identity.models import UserContext

from coreason_enclave.api import app as api_app
from coreason_enclave.federation.executor import start_client
from coreason_enclave.utils.logger import logger


def apply_security_policy(simulation_flag: bool, insecure_flag: bool) -> None:
    """
    Configure the security mode for the Enclave Agent.

    Strictly enforces the presence of the --insecure or --simulation flag for simulation mode.
    This ensures that production workloads cannot accidentally run in an untrusted environment.

    Args:
        simulation_flag (bool): True if --simulation was passed in CLI.
        insecure_flag (bool): True if --insecure was passed in CLI.

    Raises:
        RuntimeError: If simulation mode is requested via environment but the required CLI flag is missing.
    """
    env_simulation = os.environ.get("COREASON_ENCLAVE_SIMULATION", "false").lower() == "true"
    requested_simulation = simulation_flag or insecure_flag

    if requested_simulation:
        logger.warning("!!! RUNNING IN INSECURE SIMULATION MODE !!!")
        logger.warning("Hardware attestation checks are BYPASSED via --simulation/--insecure flag.")
        logger.warning("Do NOT use this mode for production data.")
        os.environ["COREASON_ENCLAVE_SIMULATION"] = "true"
    else:
        # Strict Enforcement: If environment requests simulation but flag is missing, ABORT.
        if env_simulation:
            error_msg = (
                "Security Violation: COREASON_ENCLAVE_SIMULATION=true is set in the environment, "
                "but the required '--insecure' or '--simulation' CLI flag is missing. "
                "Refusing to launch in insecure mode without explicit CLI override."
            )
            logger.critical(error_msg)
            raise RuntimeError(error_msg)

        # Force secure mode to prevent accidental insecurity from lower layers
        os.environ["COREASON_ENCLAVE_SIMULATION"] = "false"
        logger.info("Running in SECURE HARDWARE MODE. TEE Attestation required.")


def run_api_server() -> None:
    """Run the sidecar Management API server."""
    logger.info("Starting Management API on 127.0.0.1:8000")
    try:
        # Constraint: Only listen on loopback interface
        # Run programmatically to share the process/environment
        uvicorn.run(api_app, host="127.0.0.1", port=8000, log_level="info")
    except Exception as e:
        logger.error(f"Failed to start Management API: {e}")


def main(args: Optional[list[str]] = None) -> None:
    """
    Entry point for the Coreason Enclave Agent.

    Wraps NVFlare's ClientTrain to start the federation client.
    It performs the following steps:
    1. Validates security flags (Simulation vs. TEE).
    2. Starts the Management API (Sidecar) in a background thread.
    3. Translates arguments for NVFlare.
    4. Invokes the NVFlare client process.

    Args:
        args (Optional[list[str]]): Command line arguments. Defaults to sys.argv[1:].
    """
    logger.info("Starting Coreason Enclave Agent...")

    parser = argparse.ArgumentParser(description="Coreason Enclave Agent")
    parser.add_argument("--workspace", "-w", type=str, required=True, help="Path to workspace directory")
    parser.add_argument("--conf", "-c", type=str, required=True, help="Path to client config file")
    parser.add_argument("--config_folder", "-f", type=str, default="config", help="Config folder path")
    parser.add_argument(
        "--simulation",
        action="store_true",
        help="Run in simulation mode (bypasses hardware attestation)",
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Alias for --simulation",
    )
    # Allows passing arbitrary args to NVFlare
    parser.add_argument("opts", nargs=argparse.REMAINDER, help="Additional options")

    try:
        parsed_args = parser.parse_args(args)

        # 1. Validate and Configure Security
        apply_security_policy(simulation_flag=parsed_args.simulation, insecure_flag=parsed_args.insecure)

        logger.info(f"Workspace: {parsed_args.workspace}")
        logger.info(f"Config: {parsed_args.conf}")

        # 2. Start Management API (Sidecar)
        # We start this before the executor so the health check is available immediately (initializing state)
        # Using a daemon thread so it dies when the main process (NVFlare) exits.
        api_thread = threading.Thread(target=run_api_server, daemon=True)
        api_thread.start()

        # 3. Create System Context
        system_context = UserContext(
            user_id="cli-user",
            username="cli-user",
            email="cli@coreason.ai",
            permissions=["system"],
            project_context="cli",
        )

        # 4. Invoke Executor
        start_client(
            context=system_context,
            workspace=parsed_args.workspace,
            conf=parsed_args.conf,
            opts=parsed_args.opts,
        )

    except Exception as e:
        logger.exception(f"Failed to start Enclave Agent: {e}")
        sys.exit(1)


if __name__ == "__main__":  # pragma: no cover
    main()  # pragma: no cover
