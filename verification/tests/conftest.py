"""Put `verification/` on the path, so these tests can import the tooling.

These tests are deliberately NOT part of the product suite: `testpaths` is
`tests/`, and a test under `tests/` may not import anything outside it,
because CI copies `tests/` elsewhere and runs it against the installed
package. So they live here and are run by their own command.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
