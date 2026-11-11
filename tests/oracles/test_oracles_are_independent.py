"""The oracles must not quietly start sharing anything.

Each module in this layer answers a question the implementation also answers,
by a route that owes it nothing. The value of that is exactly the
independence: if a helper were shared between an oracle and the thing it
checks, a bug in the helper would make the two agree wrongly, the suite would
pass, and the passing suite would be the evidence for the bug. There is no
test that catches that afterwards -- which is why it is checked here, as
structure, before it can happen.

These tests read files only, so they run from a copy of `tests/` exactly as
they run here.
"""
import ast
import io
import re
import sys
import tokenize
from pathlib import Path

HERE = Path(__file__).resolve().parent
TESTS = HERE.parent

#: The C oracle: the hierarchical prior restated from its definition, for the
#: storage tests that check a finalized cell against the counts it came from.
C_ORACLE = TESTS / "harness" / "oracles" / "prior_reference.h"

#: The harness, which must no longer carry an oracle inside it.
C_HARNESS_HEADER = TESTS / "native" / "test_utils.h"

#: The shared engineering layer, by name. Looked for in CODE and not in prose:
#: the layer's own `__init__.py` says in its docstring that nothing here
#: imports it, and a rule that its own statement violates is a rule nobody can
#: write down. The import list is read with `ast`, and the token is then looked
#: for in the source with every comment and string removed, which still catches
#: a `sys.path` edit or an `importlib.import_module` by name.
SHARED_LAYER = "_engineering"

#: What the C oracle is allowed to call: the accessor that hands it the
#: training counts. Those are its INPUT. Everything else -- the levels, the
#: backoff, the logs -- it computes itself, and a second name appearing here
#: would mean it had started borrowing the arithmetic it exists to check.
C_ORACLE_MAY_CALL = {"beam_cnt_row"}


def _oracle_modules():
    return sorted(p for p in HERE.rglob("*.py") if not p.name.startswith("test_"))


def _code_only(path):
    """The source with every comment and string literal removed."""
    kept = []
    for token in tokenize.generate_tokens(
            io.StringIO(path.read_text(encoding="utf-8")).readline):
        if token.type not in (tokenize.COMMENT, tokenize.STRING):
            kept.append(token.string)
    return " ".join(kept)


def _import_roots(path):
    """The top-level module name of every import in `path`."""
    roots = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            roots |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def test_no_oracle_imports_the_shared_engineering():
    for path in _oracle_modules():
        assert SHARED_LAYER not in _import_roots(path), path
        assert SHARED_LAYER not in _code_only(path), path


def test_no_oracle_imports_another_oracle():
    """They must be independent of each other as well as of the library: two
    oracles sharing a helper is the same failure as an oracle sharing one
    with the implementation."""
    siblings = {p.stem for p in _oracle_modules()} - {"__init__"}
    for path in _oracle_modules():
        borrowed = (_import_roots(path) & siblings) - {path.stem}
        assert borrowed == set(), (path, borrowed)
        assert "oracles" not in _import_roots(path), path


def test_the_reference_model_imports_only_the_standard_library_and_numpy():
    outside = _import_roots(HERE / "beam_model.py") - set(sys.stdlib_module_names)
    assert outside == {"numpy"}


def test_the_c_prior_reference_reads_counts_and_computes_the_rest_itself():
    called = set(re.findall(r"\b(beam_[a-z_]+)\s*\(",
                            C_ORACLE.read_text(encoding="utf-8")))
    assert called == C_ORACLE_MAY_CALL


def test_the_c_harness_header_no_longer_carries_an_oracle():
    """It held the prior reference behind an opt-in macro, which is how a
    129-line oracle came to live in a file 35 suites include for its
    assertions."""
    text = C_HARNESS_HEADER.read_text(encoding="utf-8")
    assert "PriorRef" not in text, C_HARNESS_HEADER
    assert "BEAM_TEST_PRIOR_REFERENCE" not in text, C_HARNESS_HEADER
