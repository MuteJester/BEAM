"""The suite imports its shared engineering explicitly, or it does not run.

Three bare module names used to resolve here, for two different invisible
reasons. `from _helpers import ...` worked because `tests/conftest.py` put
`tests/python` on `sys.path`. `from _tiny import ...` and `from _brute
import ...` worked because there is no `__init__.py` under `tests/python`, so
pytest's `prepend` import mode inserted each test file's OWN directory as it
collected it -- those two were on no path entry any conftest had ever heard
of. Neither reason is legible at the import site, and neither survives
`--import-mode=importlib`, which inserts nothing at all.

These tests are in the suite rather than beside the gate, because they read
only files under `tests/` and CI runs the suite from a copy of this directory:
an import that resolves here and not there is exactly the failure they are
for.
"""
import re
from pathlib import Path

TESTS = Path(__file__).resolve().parents[1]

BARE = re.compile(r"^\s*(?:from|import)\s+(_helpers|_tiny|_brute)\b")

#: A fixtures directory reached by counting `parents[N]` from the file doing
#: the reaching means "N directories up from wherever this file happens to
#: sit", which silently becomes a different directory the moment the file
#: moves -- and moving files is what a layer structure is.
DEPTH = re.compile(r"""parents\[\d+\]\s*/\s*["']fixtures["']""")

#: The one module allowed to match it: it counts from ITSELF, and it is the
#: layer root, so the count cannot change without this file moving too.
PATH_OWNER = "_engineering/fixtures.py"


def _offenders(pattern):
    hits = []
    for path in sorted(TESTS.rglob("*.py")):
        for number, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), 1):
            if pattern.search(line):
                hits.append(f"{path.relative_to(TESTS)}:{number}")
    return hits


def test_no_test_imports_a_bare_helper_module():
    assert _offenders(BARE) == []


def test_no_fixture_path_depends_on_nesting_depth():
    outside = [hit for hit in _offenders(DEPTH)
               if not hit.startswith(PATH_OWNER)]
    assert outside == []


def test_the_fixture_directory_is_the_one_holding_the_irreplaceable_files():
    """Two `.beam` files in `tests/fixtures/` are real output from builds that
    no longer exist and cannot be regenerated. `fixtures_dir()` is the one
    place that says where they are."""
    from _engineering.fixtures import fixtures_dir

    found = fixtures_dir()
    assert found.is_dir(), found
    assert sorted(p.name for p in found.glob("*.beam")) == [
        "format6_sparse.beam", "format9_short.beam"]
