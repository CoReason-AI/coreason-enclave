import sys
from typing import List, Optional
from unittest.mock import MagicMock
from threading import Lock

import pytest
from pydantic import BaseModel


def pytest_configure(config: MagicMock) -> None:
    # Mock coreason-identity
    if "coreason_identity" not in sys.modules:
        mock_identity = MagicMock()

        # Mock UserContext class
        class MockUserContext(BaseModel):
            user_id: str
            username: Optional[str] = None
            email: Optional[str] = None
            permissions: Optional[List[str]] = None
            project_context: Optional[str] = None
            privacy_budget_spent: float = 0.0
            privacy_budget_limit: float = float("inf")

        mock_identity.models.UserContext = MockUserContext

        # Mock Exceptions
        class MockIdentityVerificationError(Exception):
            pass

        mock_identity.exceptions.IdentityVerificationError = MockIdentityVerificationError

        sys.modules["coreason_identity"] = mock_identity
        sys.modules["coreason_identity.models"] = mock_identity.models
        sys.modules["coreason_identity.exceptions"] = mock_identity.exceptions


@pytest.fixture(autouse=True)
def reset_enclave_singleton():
    """
    Reset the CoreasonEnclaveService singleton before and after each test.
    This prevents state pollution (e.g., SCAFFOLD control variates, privacy guards)
    between tests, as the service is now a singleton shared across the process.
    """
    # Import inside fixture to avoid top-level import before mocking
    from coreason_enclave.services import CoreasonEnclaveService

    # Teardown previous state if any
    if CoreasonEnclaveService._instance:
        # Best effort cleanup if portal is open
        try:
            if CoreasonEnclaveService._instance._portal:
                CoreasonEnclaveService._instance.__exit__(None, None, None)
        except Exception:
            pass
        CoreasonEnclaveService._instance = None

    # Reset lock to ensure no deadlocks from previous mocked locks
    CoreasonEnclaveService._lock = Lock()

    yield

    # Cleanup after test
    if CoreasonEnclaveService._instance:
        try:
            if CoreasonEnclaveService._instance._portal:
                CoreasonEnclaveService._instance.__exit__(None, None, None)
        except Exception:
            pass
        CoreasonEnclaveService._instance = None
