import sys
from unittest.mock import MagicMock
from pydantic import BaseModel
from typing import Optional, List

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

        mock_identity.models.UserContext = MockUserContext

        # Mock Exceptions
        class MockIdentityVerificationError(Exception):
            pass

        mock_identity.exceptions.IdentityVerificationError = MockIdentityVerificationError

        sys.modules["coreason_identity"] = mock_identity
        sys.modules["coreason_identity.models"] = mock_identity.models
        sys.modules["coreason_identity.exceptions"] = mock_identity.exceptions
