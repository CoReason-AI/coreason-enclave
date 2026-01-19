# The Architecture and Utility of coreason-enclave

### 1. The Philosophy (The Why)

In the high-stakes world of biomedical research and pharmaceutical development, data is the most valuable asset—and the most vulnerable liability. Organizations are paralyzed by a paradox: they need to collaborate on massive datasets to cure diseases, but they cannot share a single row of patient data due to privacy laws (HIPAA/GDPR) and intellectual property concerns.

**coreason-enclave** resolves this paradox by inverting the standard machine learning paradigm. Instead of moving data to the model, it **moves the model to the data**.

It acts as a digital "Embassy"—a sovereign, secure territory established inside a partner's data center. By leveraging **Trusted Execution Environments (TEEs)**, it ensures that the training process occurs within hardware-encrypted RAM. Even the host machine's operating system or a rogue root administrator cannot inspect the model's memory or the data it processes.

Combined with **Federated Learning** and **Differential Privacy**, `coreason-enclave` allows an AI model to learn like a "sightless surgeon": it operates with precision on sensitive data without ever "seeing" or remembering the individual identities of the patients. This architecture provides the mathematical and cryptographic guarantees necessary to unlock "pre-competitive" collaboration among fierce rivals.

### 2. Under the Hood (The Dependencies & logic)

The package is built on a stack of state-of-the-art libraries, each serving a critical role in the "Defense in Depth" strategy:

*   **`nvflare` (NVIDIA FLARE):** The backbone of the federation. It handles the complex choreography of distributing the global model, orchestrating training rounds, and aggregating updates. It allows `coreason-enclave` to act as a compliant node in a distributed network.
*   **`opacus` (PyTorch):** The privacy engine. While TEEs protect data *during* computation, `opacus` ensures the *output* (the model weights) cannot be reverse-engineered. It implements Differential Privacy (DP-SGD) by clipping gradients and injecting statistical noise, rigorously tracking the "privacy budget" ($\epsilon$) consumed.
*   **`torch` (PyTorch):** The deep learning framework. `coreason-enclave` is framework-agnostic in principle but optimized for PyTorch's ecosystem.
*   **`pydantic`:** The rigid enforcer. It validates every configuration and data schema, ensuring that malformed or malicious inputs are rejected before they touch the training logic.

**The "Security Sandwich" Logic:**
The internal architecture follows a strict execute flow known as the "Security Sandwich":
1.  **Attestation:** The `AttestationProvider` first cryptographically proves the hardware's integrity.
2.  **Input Airlock (`DataSentry`):** Data is loaded only if it passes strict path validation and schema checks.
3.  **Privacy-Preserved Training:** The `CoreasonExecutor` runs the training loop. The `PrivacyGuard` wraps the optimizer, enforcing gradient clipping and noise injection on every batch.
4.  **Output Airlock (`DataSentry`):** Before any result leaves the enclave, the `DataSentry` sanitizes the payload, stripping unauthorized keys and ensuring no raw data leaks.

### 3. In Practice (The How)

In practice, `coreason-enclave` is designed to be deployed invisibly, managed by the orchestration layer. However, its internal API reveals the elegance of its security model.

#### The Orchestrator (`CoreasonExecutor`)
The executor is the heart of the package. It listens for tasks, verifies the hardware trust, and manages the lifecycle of a training round.

```python
from coreason_enclave.federation.executor import CoreasonExecutor

# Initialize the executor, which automatically sets up
# the hardware attestation and data sentries.
executor = CoreasonExecutor(training_task_name="train_model")

# The executor runs within the NVFlare cycle, but here is the
# logical flow of a single task execution:
def run_secure_task(shareable, fl_context):
    # 1. Hardware Verify: Ensures we are running in a genuine TEE
    # Raises RuntimeError if the environment is untrusted
    executor._check_hardware_trust()

    # 2. Execute: Parses the job, loads the data through the Sentry,
    # and runs the privacy-preserving training loop.
    result_shareable = executor.execute(
        task_name="train_model",
        shareable=shareable,
        fl_ctx=fl_context,
        abort_signal=None
    )

    return result_shareable
```

#### The Privacy Guard
The `PrivacyGuard` ensures that the model training strictly adheres to the privacy budget. It wraps the standard PyTorch optimizer, making the privacy enforcement transparent to the model architecture.

```python
from coreason_enclave.privacy import PrivacyGuard
from coreason_enclave.schemas import PrivacyConfig

# Define the privacy strictness (Epsilon < 5.0 is standard)
config = PrivacyConfig(
    target_epsilon=3.0,
    noise_multiplier=1.0,
    max_grad_norm=1.0
)

guard = PrivacyGuard(config)

# Attach privacy mechanisms to the standard PyTorch components
# This transforms the optimizer into a differentially private one
model, optimizer, loader = guard.attach(
    model=standard_model,
    optimizer=standard_optimizer,
    data_loader=standard_loader
)

# Training proceeds normally, but gradients are now noisy and clipped
optimizer.step()

# Check budget consumption - throws an exception if we learned "too much"
guard.check_budget()
```
