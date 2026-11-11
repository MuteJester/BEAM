"""The suite imports its shared engineering explicitly, or it does not run.

Three bare module names used to resolve here, for two different invisible
reasons. `from _helpers import ...` worked because `tests/conftest.py` put
the suite directory on `sys.path`. `from _tiny import ...` and `from _brute
import ...` worked because there is no `__init__.py` under the suites, so
pytest's `prepend` import mode inserted each test file's OWN directory as it
collected it -- those two were on no path entry any conftest had ever heard
of. Neither reason is legible at the import site, and neither survives
`--import-mode=importlib`, which inserts nothing at all.

These tests are in the suite rather than beside the gate, because they read
only files under `tests/` and CI runs the suite from a copy of this directory:
an import that resolves here and not there is exactly the failure they are
for.
"""
import ast
import re
from pathlib import Path

TESTS = Path(__file__).resolve().parents[1]

BARE = re.compile(r"^\s*(?:from|import)\s+(_helpers|_tiny|_brute)\b")

#: The layers. A module INSIDE one of them is reachable only through it, so
#: the import line names the layer and nothing resolves by accident.
LAYERS = ("_engineering", "oracles")

#: A fixtures directory reached by counting `parents[N]` from the file doing
#: the reaching means "N directories up from wherever this file happens to
#: sit", which silently becomes a different directory the moment the file
#: moves -- and moving files is what a layer structure is.
DEPTH = re.compile(r"""parents\[\d+\]\s*/\s*["']fixtures["']""")

#: The one module allowed to match it: it counts from ITSELF, and it is the
#: layer root, so the count cannot change without this file moving too.
PATH_OWNER = "_engineering/fixtures.py"


def _modules_of_the_layers():
    """Every module name that belongs to a layer: `corpora`, `fixtures`,
    `mixture`, `likelihood`, `beam_model`. Read from the tree rather than
    listed, so a module added to a layer is covered the day it arrives --
    the previous version of this test named three modules by hand and missed
    a second import of a fourth on the line below the one it checked."""
    return {path.stem
            for layer in LAYERS
            for path in (TESTS / layer).rglob("*.py")} - {"__init__"}


def _import_roots(path):
    """The top-level name of every import in `path`, including the ones
    inside functions: one of the three bare imports was in a function body."""
    roots = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            roots |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def _is_the_path_owner(hit):
    """True for a hit in the one module allowed to compute the fixture path.

    Matched as a POSIX path. `Path.relative_to` renders with the platform's
    separator, so on Windows the owner's own hit came back as
    `_engineering\\fixtures.py` and did not match its forward-slash
    exemption -- the module was reported as an offender against itself, and
    both Windows legs of CI failed on it while all six other legs passed.
    """
    return hit.replace("\\", "/").startswith(PATH_OWNER)


def _offenders(pattern):
    hits = []
    for path in sorted(TESTS.rglob("*.py")):
        for number, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), 1):
            if pattern.search(line):
                # as_posix: these strings are COMPARED against
                # PATH_OWNER below, and Windows renders a relative
                # path with backslashes, so the one module allowed to
                # compute the fixture path stopped matching its own
                # exemption and the suite failed on both Windows legs.
                hits.append(f"{path.relative_to(TESTS).as_posix()}:{number}")
    return hits


def test_no_test_imports_a_bare_helper_module():
    inside = _modules_of_the_layers()
    bare = sorted(f"{path.relative_to(TESTS).as_posix()}: {sorted(found)}"
                  for path in sorted(TESTS.rglob("*.py"))
                  if path.parent.name not in LAYERS
                  for found in [_import_roots(path) & inside] if found)
    assert bare == []
    assert _offenders(BARE) == []


def test_no_fixture_path_depends_on_nesting_depth():
    # Both spellings of the owner's own path, because this ran green on
    # Linux and failed on Windows for the separator alone.
    assert _is_the_path_owner("_engineering/fixtures.py:41")
    assert _is_the_path_owner("_engineering\\fixtures.py:41")

    outside = [hit for hit in _offenders(DEPTH) if not _is_the_path_owner(hit)]
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
