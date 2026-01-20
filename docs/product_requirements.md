# Product Requirements Document: coreason-enclave

**Domain:** Confidential Computing, Federated Learning (FL), & Differential Privacy
**Architectural Role:** The "Embassy" / The Secure Compute Wrapper, hardware-encrypted **Trusted Execution Environments (TEEs)**
**Core Philosophy:** "Move the Model to the Data. Never move the Data. Encrypt the RAM."
**Dependencies:** coreason-model-foundry (Caller), coreason-identity (Attestation), nvflare (Federation), opacus (Differential Privacy)

## 1. Executive Summary

coreason-enclave is the privacy-preserving compute layer. It allows coreason-model-foundry to orchestrate training jobs across distributed nodes (e.g., multiple hospitals or partner pharma companies) without accessing their raw data.

It combines three SOTA technologies:

1.  **Federated Learning (FL):** Training local models on private data and only sharing weight updates (gradients), not records.
2.  **Confidential Computing (TEEs):** Running the training inside hardware-encrypted enclaves (e.g., NVIDIA H100 Confidential Compute), ensuring that even the cloud provider (AWS/Azure) cannot inspect the memory.
3.  **Differential Privacy (DP):** Adding mathematical noise to the gradients to guarantee that no single patient's data can be reverse-engineered from the model updates.

## 2. Functional Philosophy

The agent must implement the **Attest-Train-Aggregrate Loop**:

1.  **Remote Attestation (The Handshake):** Before a Hospital allows CoReason to run code on its server, enclave must cryptographically prove: *"I am running the exact, signed version of the training script. My RAM is encrypted by an Intel SGX/NVIDIA H100 enclave."*
2.  **The "Sightless" Surgeon:** The AI acts like a surgeon operating in the dark. It learns from the data but never "sees" it. We use **NVIDIA FLARE** to manage this blind choreography.
3.  **Budgeted Privacy:** Privacy isn't binary; it's a budget ($\epsilon$). Every training round consumes a bit of privacy. enclave tracks the **Privacy Budget** and aborts training before the risk of re-identification exceeds the threshold.

## 3. Core Functional Requirements (Component Level)

### 3.1 The Federation Overseer (The Orchestrator)

**Concept:** Manages the global training state.

*   **Mechanism:** Wraps **NVIDIA FLARE**.
*   **Role:**
    *   Distributes the "Global Model" to all participating Clients (Hospitals/Labs).
    *   Orchestrates "Rounds" (Train $\to$ Upload Gradients $\to$ Aggregate $\to$ Redistribute).
    *   **Aggregation Strategy:** Implements **FedAvg** (Federated Averaging) or **FedProx** (Robustness to non-IID data).

### 3.2 The Enclave Wrapper (The Container)

**Concept:** Wraps the training code in a TEE.

*   **Technology:** Uses **Gramine** or **SCONE** to wrap the Python training script into an SGX/SEV-SNP enclave.
*   **Function:** Ensures that memory pages are encrypted at the CPU level. Even if a hacker has root access to the server, the RAM reads as static noise.
*   **Key Feature:** **Remote Attestation**. Generates a hardware-signed quote proving the code's integrity.

### 3.3 The Privacy Guard (The Noise Injector)

**Concept:** Implements Differential Privacy (DP-SGD).

*   **Library:** **Opacus** (PyTorch).
*   **Action:**
    *   Clips the gradients (limiting the impact of outliers).
    *   Adds Gaussian noise to the gradients *before* they leave the secure enclave.
*   **Metric:** Tracks **Epsilon ($\epsilon$)**. If $\epsilon > 5.0$, the system halts to prevent privacy leakage.

### 3.4 The Data Sentry (The Firewall)

**Concept:** Input/Output sanitation for the enclave.

*   **Input:** Validates that the local data matches the expected schema (using coreason-validator).
*   **Output:** "Airlock." Ensures no raw data leaks out in the logs. Only model weights and loss metrics are allowed to exit the enclave.

## 4. Integration Requirements

*   **coreason-model-foundry:**
    *   Foundry defines *what* to train (model_def.py).
    *   Enclave defines *where* and *how* (Federated + Secure).
    *   Foundry submits a job to enclave.orchestrate(job_def).
*   **coreason-identity:**
    *   Manages the "Federation Trust Root."
    *   Verifies the Attestation Quotes from the hardware to ensure no malicious nodes have joined the network.
*   **coreason-veritas:**
    *   Logs the **Hash** of every code update sent to the federation.
    *   Logs the **Privacy Budget** consumed per round.

## 5. User Stories

### Story A: The "Pre-Competitive" Safety Model

**Context:** Pharma A, B, and C want to build a better model to predict Liver Toxicity. None wants to share their proprietary compound structures.
**Action:** coreason-enclave establishes a Federation.
**Process:**

1.  The model travels to Pharma A, trains on their secure enclave.
2.  The *encrypted gradients* travel to the Aggregator.
3.  Repeats for Pharma B and C.
**Result:** A SOTA Toxicity model is created. No IP ever left any company's firewall.

### Story B: The "Hospital Firewall" (RWD)

**Context:** We need to fine-tune a clinical trial matching agent on real patient records at Mayo Clinic.
**Constraint:** Patient data cannot leave the hospital (HIPAA).
**Action:** Deploy an enclave agent to the Hospital's secure zone.
**Attestation:** Hospital IT verifies the enclave signature. "Okay, this code is trusted."
**Training:** Model trains locally.
**Output:** Only the weight updates are sent back to CoReason.

### Story C: The "Secure Inference" (Consulting)

**Context:** A client wants to use CoReason's model but has a Molecule X that is worth $10B. They are terrified of leaking the structure to the cloud.
**Action:** Client submits Molecule X into an H100 Confidential Compute instance running enclave.
**Guarantee:** Not even CoReason admins can see the prompt inside the enclave.
**Result:** Client gets the prediction. CoReason gets paid. The molecule remains secret.

## 6. Data Schema

### FederationConfig

```python
class AggregationStrategy(str, Enum):
    FED_AVG = "FED_AVG"         # Standard averaging
    FED_PROX = "FED_PROX"       # Handles non-IID data
    SCAFFOLD = "SCAFFOLD"       # Controls client drift

class PrivacyConfig(BaseModel):
    mechanism: str = "DP_SGD"   # Differential Privacy
    noise_multiplier: float     # 1.0
    max_grad_norm: float        # 1.0
    target_epsilon: float       # 3.0 (Strict privacy)

class FederationJob(BaseModel):
    job_id: UUID
    clients: List[str]          # ["node_hospital_a", "node_hospital_b"]
    min_clients: int            # 2
    rounds: int                 # 50
    strategy: AggregationStrategy
    privacy: PrivacyConfig
```

### AttestationReport

```python
class AttestationReport(BaseModel):
    node_id: str
    hardware_type: str          # "NVIDIA_H100_HOPPER"
    enclave_signature: str      # The hardware quote
    measurement_hash: str       # SHA256 of the running binary
    status: Literal["TRUSTED", "UNTRUSTED"]
```

## 7. Implementation Directives for the Coding Agent

1.  **Framework Selection (SOTA):**
    *   Use **NVIDIA FLARE** (NVFlare). It is the industrial standard for Medical Imaging and Bio-Pharma FL. It supports secure provisioning and integrates with Confidential Compute.
2.  **Privacy Library:**
    *   Use **Opacus** for PyTorch. It handles the mathematical complexity of gradient clipping and noise addition automatically.
3.  **Simulation Mode:**
    *   Real TEE hardware (H100s) is expensive. Implement a simulation_mode: bool that mimics the federation locally (using threads/processes) for developer testing without requiring special hardware.
4.  **Hardware Binding:**
    *   When running in production, explicitly check for **SNP/TDX/SGX** capabilities. If the hardware does not support encryption, enclave must refuse to launch unless explicitly overridden with a --insecure flag.
