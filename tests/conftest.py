"""Make the test helpers and the reference implementation importable.

The suite is split into tests/python/<domain>/, with shared fixtures in
tests/python/conftest.py, constants in tests/python/_helpers.py, and the
independent reference in tests/reference/. Those last two are plain modules
rather than fixtures, so their directories go on the path here.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for d in (ROOT / "python", ROOT / "reference"):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))
