# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_enclave

import argparse
import os
import sys
from typing import List, Optional

# Workaround for NVFlare Windows issue
if sys.platform == "win32":  # pragma: no cover
    from unittest.mock import MagicMock

    try:
        import resource  # noqa: F401
    except ImportError:
        sys.modules["resource"] = MagicMock()

try:
    # Try importing NVFlare client logic
    import nvflare.private.fed.app.client.client_train as client_train
except ImportError:
    # Fallback for environments without nvflare installed (e.g. specialized test envs)
    # or if the internal path changes.
    client_train = None

from coreason_enclave.utils.logger import logger


def apply_security_policy(simulation_flag: bool, insecure_flag: bool) -> None:
    """
    Configure the security mode for the Enclave Agent.
    Strictly enforces the presence of the --insecure or --simulation flag for simulation mode.

    Args:
        simulation_flag: True if --simulation was passed in CLI.
        insecure_flag: True if --insecure was passed in CLI.

    Raises:
        RuntimeError: If simulation mode is requested via environment but flag is missing.
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


def _build_nvflare_args(parsed_args: argparse.Namespace) -> List[str]:
    """
    Construct the argument list for NVFlare ClientTrain.

    Args:
        parsed_args: The arguments parsed by coreason-enclave.

    Returns:
        List[str]: The argument list for nvflare.
    """
    # NVFlare client_train.py expects:
    # --workspace (-m) <workspace>
    # --fed_client (-s) <client config file>
    # --set KEY=VALUE ...
    # --local_rank <int>

    # coreason-enclave args: -w/--workspace, -c/--conf
    # Mapping:
    # -w -> -m (workspace)
    # -c -> -s (fed_client / config)

    args = [
        "coreason_enclave_wrapper",  # Prog name for argv[0]
        "-m",
        parsed_args.workspace,
        "-s",
        parsed_args.conf,
    ]

    # Pass through other options if they match NVFlare structure
    if parsed_args.opts:
        args.extend(parsed_args.opts)

    return args


def main(args: Optional[list[str]] = None) -> None:
    """
    Entry point for the Coreason Enclave Agent.
    Wraps NVFlare's ClientTrain to start the federation client.
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

        # 2. Construct NVFlare arguments
        nvflare_argv = _build_nvflare_args(parsed_args)

        # 3. Invoke NVFlare Client
        if client_train:
            logger.info("Invoking NVFlare ClientTrain...")

            # We must manipulate sys.argv because client_train.parse_arguments() reads it directly.
            sys_argv_backup = sys.argv
            try:
                sys.argv = nvflare_argv
                # Parse args using NVFlare's parser
                fl_args = client_train.parse_arguments()
                # Run main logic with parsed args
                client_train.main(fl_args)
            finally:
                sys.argv = sys_argv_backup
        else:
            logger.warning("NVFlare ClientTrain module not found. Skipping execution (Dry Run).")

    except Exception as e:
        logger.exception(f"Failed to start Enclave Agent: {e}")
        sys.exit(1)


if __name__ == "__main__":  # pragma: no cover
    main()  # pragma: no cover
