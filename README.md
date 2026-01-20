# coreason-enclave

[![License: Prosperity 3.0](https://img.shields.io/badge/license-Prosperity%203.0-blue)](https://github.com/CoReason-AI/coreason-enclave)
[![CI Status](https://github.com/CoReason-AI/coreason-enclave/actions/workflows/ci.yml/badge.svg)](https://github.com/CoReason-AI/coreason-enclave/actions)
[![Code Style: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/CoReason-AI/coreason-enclave)
[![Documentation](https://img.shields.io/badge/docs-PRD-informational)](docs/product_requirements.md)

**Privacy-Preserving Compute Layer for Federated Learning**

**coreason-enclave** is the "Embassy" / Secure Compute Wrapper of the CoReason AI ecosystem. It acts as the privacy-preserving compute layer, allowing orchestration of training jobs across distributed nodes (e.g., multiple hospitals or partner pharma companies) without accessing their raw data. It combines Federated Learning, Confidential Computing (TEEs), and Differential Privacy to ensure a mathematical guarantee of privacy.

> **Core Philosophy:** "Move the Model to the Data. Never move the Data. Encrypt the RAM."

---

## 🚀 Features

*   **Federated Learning (FL):** Orchestrate training across distributed nodes using **NVIDIA FLARE**. Only weight updates (gradients) are shared, never raw data. Supports FedAvg, FedProx, and SCAFFOLD strategies.
*   **Confidential Computing:** Designed to run inside hardware-encrypted **Trusted Execution Environments (TEEs)** (e.g., NVIDIA H100 Confidential Compute, Intel SGX). This ensures memory is encrypted at the CPU level, protecting against cloud provider inspection. Includes **Remote Attestation** to cryptographically prove code integrity.
*   **Differential Privacy (DP):** Integrated with **Opacus** to inject Gaussian noise into gradients, strictly enforcing a privacy budget ($\epsilon$).
*   **The "Sightless" Surgeon:** The AI learns from data it never "sees."
*   **Data Sentry:** An "Airlock" mechanism that validates input data and strictly sanitizes output, ensuring no sensitive information leaks via logs or return payloads.

## 🛠️ Installation

```bash
pip install coreason-enclave
```

## 💻 Usage

The `coreason-enclave` agent typically runs as a service managed by an orchestrator, but can be invoked directly or integrated into custom workflows.

### Basic Initialization

```python
from coreason_enclave.federation.executor import CoreasonExecutor
from coreason_enclave.schemas import FederationJob

# Initialize the Executor
executor = CoreasonExecutor(
    training_task_name="train",
    aggregation_task_name="aggregate"
)

# Note: In production, this is handled automatically by the NVFlare runtime.
# The executor listens for tasks from the Federation Overseer.
```

### Running the Agent (CLI)

To start the agent as a standalone client connecting to a federation:

```bash
# Secure Mode (Requires TEE Hardware)
python -m coreason_enclave.main \
    --workspace /tmp/workspace \
    --conf config/client_config.json

# Simulation Mode (For Development/Testing)
python -m coreason_enclave.main \
    --workspace /tmp/workspace \
    --conf config/client_config.json \
    --simulation
```

For more detailed requirements and architecture, please refer to the [Product Requirements Document](docs/product_requirements.md).

## 📜 License

This project is licensed under the **Prosperity Public License 3.0**.
See the [LICENSE](LICENSE) file for details.
