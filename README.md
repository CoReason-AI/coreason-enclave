# coreason-enclave

[![License: Prosperity 3.0](https://img.shields.io/badge/License-Prosperity%203.0-blue)](https://github.com/CoReason-AI/coreason-enclave)
[![CI Status](https://github.com/CoReason-AI/coreason-enclave/actions/workflows/ci.yml/badge.svg)](https://github.com/CoReason-AI/coreason-enclave/actions)
[![Code Style: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/CoReason-AI/coreason-enclave)

**Privacy-Preserving Compute Layer for Federated Learning**

`coreason-enclave` is the "Embassy" of the CoReason AI ecosystem. It acts as a secure compute wrapper that allows training of machine learning models on distributed, private data without ever exposing that data. It combines Federated Learning, Confidential Computing (TEEs), and Differential Privacy to ensure a mathematical guarantee of privacy.

---

## 🚀 Features

*   **Federated Learning (FL):** Orchestrate training across distributed nodes using **NVIDIA FLARE**. Only weight updates (gradients) are shared, never raw data.
*   **Confidential Computing:** Designed to run inside hardware-encrypted **Trusted Execution Environments (TEEs)** (e.g., NVIDIA H100 Confidential Compute, Intel SGX). This ensures memory is encrypted at the CPU level, protecting against cloud provider inspection.
*   **Differential Privacy (DP):** Integrated with **Opacus** to inject Gaussian noise into gradients, strictly enforcing a privacy budget ($\epsilon$).
*   **The "Sightless" Surgeon:** The AI learns from data it never "sees."
*   **Data Sentry:** An "Airlock" mechanism that validates input data and strictly sanitizes output, ensuring no sensitive information leaks via logs or return payloads.
*   **Remote Attestation:** Cryptographically proves that the running code is the exact, signed version expected by the federation.

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

## 🏗️ Architecture

*   **Overseer (Orchestrator):** Manages global training state via NVFlare.
*   **Enclave Wrapper:** Wraps training code in a TEE.
*   **Privacy Guard:** Manages the Privacy Budget ($\epsilon$) using Opacus.
*   **Data Sentry:** Input/Output firewall preventing data leakage.

## 📜 License

This project is licensed under the **Prosperity Public License 3.0**.
See the [LICENSE](LICENSE) file for details.
