"""Turn a half-installed environment into an actionable message.

Every script here imports duckdb/numpy at module scope, so running one in an
environment that lacks them answers with a raw ModuleNotFoundError traceback.
That is the one failure a reader who did not write this cannot act on, and an
activated-but-empty .venv is the usual way to land in it. atlas.py already
knows how to phrase this properly, including whether a venv is active, so
this just borrows that rather than restating it in six places.

    import _require    # noqa: F401   -- must precede duckdb/numpy imports
"""

import os
import sys

try:
    import duckdb        # noqa: F401
    import numpy         # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        from atlas import require_deps
    except Exception:                                       # noqa: BLE001
        raise SystemExit("ATLAS cannot start: its dependencies are not "
                         "installed.\n    python -m pip install -r "
                         "requirements.txt")
    raise SystemExit(f"ATLAS cannot start: {require_deps()}")
