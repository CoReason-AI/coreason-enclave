# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_enclave

import importlib
import sys
from unittest.mock import patch


def test_main_import_error_handling() -> None:
    """Test that ImportError is handled gracefully when importing nvflare."""

    # Ensure module is loaded first so we can reload it
    # Check if coreason_enclave.main is the function or module.
    # If __init__.py does `from .main import main`, then accessing `coreason_enclave.main`
    # might give the function if accessed via `coreason_enclave`.
    # But `import coreason_enclave.main` should bind the module to `coreason_enclave.main` locally
    # IF `coreason_enclave` is not already imported with the shadowed attribute.

    # Safest way: get module from sys.modules
    module_name = "coreason_enclave.main"
    if module_name not in sys.modules:
        importlib.import_module(module_name)

    module = sys.modules[module_name]

    # Mock sys.modules to simulate nvflare being missing during import
    # This now affects coreason_enclave.federation.executor
    module_name = "coreason_enclave.federation.executor"
    if module_name not in sys.modules:
        importlib.import_module(module_name)

    module = sys.modules[module_name]

    with patch.dict(sys.modules, {"nvflare.private.fed.app.client.client_train": None}):
        # Reload the module to trigger the top-level import block again
        importlib.reload(module)

        # Verify that client_train is None
        assert module.client_train is None

    # Reload again to restore normal state for other tests
    importlib.reload(module)
    assert module.client_train is not None
