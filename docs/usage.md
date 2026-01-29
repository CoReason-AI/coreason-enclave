# Usage Guide

The `coreason-enclave` acts as the secure compute layer for the CoReason ecosystem. Version 0.3.0 introduces a Service-Oriented architecture with a sidecar Health & Metrics API.

## Architecture

The Enclave Agent now runs as a dual-process application:
1.  **NVFlare Client (Main Process):** Handles the Federation logic, connecting to the Overseer, receiving models, and performing training.
2.  **Management API (Sidecar Thread):** A local FastAPI server (bound to `127.0.0.1`) that provides real-time health checks, attestation reports, and privacy budget monitoring.

Both components share a singleton `CoreasonEnclaveService` instance to ensure state consistency (e.g., Privacy Budget usage is tracked globally).

## Running the Agent

### Standard Mode (Secure)

When running on TEE hardware (e.g., NVIDIA H100 Confidential Compute):

```bash
python -m coreason_enclave.main \
    --workspace /opt/workspace \
    --conf config/client_config.json
```

The agent will:
1.  Perform hardware attestation.
2.  Start the Management API on `http://127.0.0.1:8000`.
3.  Connect to the Federation.

### Simulation Mode (Dev)

For local development without TEE hardware:

```bash
python -m coreason_enclave.main \
    --workspace /tmp/workspace \
    --conf config/client_config.json \
    --simulation
```

## Data Sentry API

The sidecar API provides the following endpoints for monitoring and verification.

### Health Check
**GET** `/health`

Returns `200 OK` if the enclave is **TRUSTED** and healthy.

```json
{
  "status": "ATTESTED",
  "trusted": true
}
```

### Hardware Attestation
**GET** `/attestation`

Returns the cryptographic proof of the enclave's integrity.

```json
{
  "node_id": "node-1",
  "hardware_type": "NVIDIA_H100_HOPPER",
  "enclave_signature": "...",
  "measurement_hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
  "status": "TRUSTED"
}
```

### Privacy Budget
**GET** `/privacy/budget`

Returns the current differential privacy budget ($\epsilon$) consumed.

```json
{
  "epsilon": 1.25
}
```

## Integration with Middleware

The CoReason Middleware (MACO) uses the API to verifying the enclave before dispatching sensitive jobs.

1.  **Poll `/health`**: Ensure the sidecar is up.
2.  **Verify `/attestation`**: Check the signature against the `coreason-identity` ledger.
3.  **Monitor `/privacy/budget`**: Ensure the budget allows for further training rounds.
