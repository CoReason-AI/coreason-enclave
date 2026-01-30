from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from coreason_enclave.schemas import AttestationReport
from coreason_enclave.services import CoreasonEnclaveService, EnclaveStatus
from coreason_enclave.utils.logger import logger


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Manage the lifecycle of the Enclave API and Service.

    Ensures:
    1. Service singleton is initialized.
    2. Context manager (BlockingPortal) is active.
    3. Initial Hardware Attestation is performed successfully.
    """
    # Initialize Service Singleton
    service = CoreasonEnclaveService.get_instance()

    # Start service context (attestation, etc.)
    # Increases ref_count for the shared singleton
    service.__enter__()

    try:
        # Perform Initial Handshake (Hardware Trust)
        logger.info("Performing initial hardware attestation...")
        try:
            report = service.refresh_attestation()
            if report.status != "TRUSTED":
                logger.critical(f"Initial attestation failed: {report.status}")
                raise RuntimeError(f"Hardware not trusted: {report.status}")
            logger.info("Enclave successfully attested and trusted.")
        except Exception as e:
            logger.critical(f"Startup attestation error: {e}")
            raise RuntimeError(f"Startup attestation error: {e}") from e

        yield
    finally:
        logger.info("Shutting down Enclave API...")
        service.__exit__(None, None, None)


app = FastAPI(title="Coreason Enclave API", lifespan=lifespan)


class PrivacyBudgetResponse(BaseModel):
    epsilon: float


class HealthResponse(BaseModel):
    status: EnclaveStatus
    trusted: bool


@app.get("/attestation", response_model=AttestationReport)  # type: ignore[misc]
async def get_attestation() -> AttestationReport:
    """
    Get the current Hardware Attestation Report.

    Verifies the TEE status cryptographically.
    """
    service = CoreasonEnclaveService.get_instance()
    try:
        return service.refresh_attestation()
    except Exception as e:
        logger.error(f"Attestation refresh failed: {e}")
        raise HTTPException(status_code=500, detail="Attestation failed") from e


@app.get("/privacy/budget", response_model=PrivacyBudgetResponse)  # type: ignore[misc]
async def get_privacy_budget() -> PrivacyBudgetResponse:
    """
    Get the current differential privacy budget (epsilon) consumed.
    """
    service = CoreasonEnclaveService.get_instance()
    return PrivacyBudgetResponse(epsilon=service.get_privacy_budget())


@app.get("/health", response_model=HealthResponse)  # type: ignore[misc]
async def get_health() -> HealthResponse:
    """
    Health check endpoint.

    Returns 200 OK only if the hardware remains in a TRUSTED state.
    """
    service = CoreasonEnclaveService.get_instance()

    current_status = service.status

    if current_status == EnclaveStatus.ERROR:
        raise HTTPException(status_code=503, detail="Enclave in ERROR state")

    if current_status == EnclaveStatus.INITIALIZING:
        raise HTTPException(status_code=503, detail="Enclave initializing")

    # If we are here, we assume trusted based on state transitions
    # (State only goes to ATTESTED/TRAINING/IDLE if attested)
    return HealthResponse(status=current_status, trusted=True)
