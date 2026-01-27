# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_enclave

import os
import sys
from unittest.mock import MagicMock

# Inject mocks into path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "mocks"))

# NVFlare 2.7.1 has an issue on Windows where it imports 'resource' (Unix-only).
# We mock it here to allow test collection to proceed on Windows.
if sys.platform == "win32":
    try:
        import resource  # type: ignore # noqa: F401
    except ImportError:
        sys.modules["resource"] = MagicMock()
