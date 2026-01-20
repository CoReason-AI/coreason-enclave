# The Architecture and Utility of coreason-enclave

### 1. The Philosophy (The Why)

In the landscape of modern AI, we face a paradox: data is most valuable when shared, but most dangerous when exposed. Traditional centralized training requires moving raw data—patient records, financial ledgers, proprietary molecules—into a central cloud, creating massive liability and regulatory nightmares.

**coreason-enclave** resolves this by inverting the paradigm. Its philosophy is simple: **"Move the Model to the Data. Never move the Data. Encrypt the RAM."**

Think of this package as a digital "Embassy." When you deploy `coreason-enclave` into a partner's infrastructure (e.g., a hospital or a competitor's lab), it creates a sovereign, secure territory. It acts as a "sightless surgeon"—it can operate on the data to learn from it, but it cannot see, copy, or leak the raw records. By combining **Federated Learning** (collaborative training), **Confidential Computing** (hardware-level memory encryption), and **Differential Privacy** (mathematical noise injection), it allows organizations to collaborate on sensitive problems without ever trusting each other with their secrets.

### 2. Under the Hood (The Dependencies & logic)

The package is built on a "Trust No One" architecture, leveraging a best-in-class stack to enforce its security guarantees:

*   **NVIDIA FLARE (`nvflare`)**: This serves as the nervous system. It handles the complex choreography of the federation—distributing global models, coordinating rounds, and aggregating updates—without ever touching the training data directly.
*   **Opacus (`opacus`)**: The mathematical conscience of the system. It wraps the PyTorch training loop to enforce Differential Privacy (DP-SGD). It clips gradients and injects statistical noise, ensuring that the model memorizes patterns, not individuals.
*   **PyTorch (`torch`)**: The deep learning engine that executes the actual training tasks.
*   **Pydantic (`pydantic`)**: The strict gatekeeper. It enforces rigorous contracts for every job configuration and attestation report, rejecting any malformed or malicious inputs before they reach the execution layer.

**The "Security Sandwich" Logic:**
The heart of the library is the `CoreasonExecutor`. It wraps the standard training loop in a rigid security protocol:
1.  **Attest:** Before lifting a finger, it queries the hardware (Intel SGX, NVIDIA H100) to cryptographically prove it is running inside an encrypted enclave.
2.  **Validate:** The `DataSentry` inspects inputs to ensure path safety and schema compliance.
3.  **Train:** The model learns in the dark, restricted by the `PrivacyGuard` which monitors the privacy budget ($\epsilon$).
4.  **Sanitize:** The "Airlock" mechanism inspects the output, stripping everything except the specific weight updates allowed to leave the enclave.

### 3. In Practice (The How)

Here is how `coreason-enclave` orchestrates this secure ballet.

**Defining the Mission**
The `FederationJob` acts as the immutable instructions for the remote agent. It dictates not just *what* to train, but the strict privacy budget allowed for the task.

```python
from coreason_enclave.schemas import FederationJob, PrivacyConfig, AggregationStrategy

# The command center defines the constraints
mission_config = FederationJob(
    job_id="liver-toxicity-study-001",
    # The agent will refuse to run if it sees fewer than 3 hospitals
    min_clients=3,
    rounds=50,
    strategy=AggregationStrategy.FED_PROX,
    # Strict privacy controls
    privacy=PrivacyConfig(
        mechanism="DP_SGD",
        target_epsilon=3.0,  # High privacy requirement
        noise_multiplier=1.0,
        max_grad_norm=1.0
    ),
    dataset_id="secure/data/liver_scans_v1",
    model_arch="SimpleMLP"
)
```

**The Secure Executor**
On the client side (e.g., inside the hospital's secure zone), the `CoreasonExecutor` receives these instructions. It establishes trust with the hardware and then executes the training loop. Notice how the `PrivacyGuard` automatically wraps the model and optimizer.

```python
from coreason_enclave.federation.executor import CoreasonExecutor
from coreason_enclave.privacy import PrivacyGuard
import torch.optim as optim

class SecureAgent(CoreasonExecutor):
    def _execute_training(self, shareable, fl_ctx, abort_signal):
        # 1. The Handshake: Cryptographically verify we are in a TEE
        self._check_hardware_trust()

        # 2. Parse the mission config from the server
        job_config = self._parse_job_config(shareable)

        # 3. Initialize Privacy Guard (The "Conscience")
        privacy_guard = PrivacyGuard(config=job_config.privacy)

        # 4. Wrap the model and optimizer for DP-SGD
        # Opacus will now intercept every gradient step to clip and noise it
        model, optimizer, train_loader = privacy_guard.attach(
            model=self.model,
            optimizer=optim.SGD(self.model.parameters(), lr=0.01),
            data_loader=self.loader
        )

        # 5. Train safely
        for epoch in range(job_config.rounds):
            for data, target in train_loader:
                optimizer.zero_grad()
                output = model(data)
                loss = self.criterion(output, target)
                loss.backward()
                optimizer.step()

                # If we burn through our privacy budget, the agent self-terminates
                privacy_guard.check_budget()

        # 6. Sanitize and return ONLY the weights
        return self.sentry.sanitize_output({
            "params": model.state_dict(),
            "metrics": {"epsilon": privacy_guard.get_current_epsilon()}
        })
```

**The Result**
The centralized server receives updated weights that are mathematically guaranteed not to leak information about any single patient, verified to come from a trusted hardware enclave. The data never moved; the model learned; and privacy was preserved.
