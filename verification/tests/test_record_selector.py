"""Which baselines `--record` writes.

`--record TAG` writes every baseline in one call, and only `.venv/bin/python` can
import BEAM here -- which is Python 3.14, where recording `python_api.txt`
would bake 3.14-only signatures into a baseline CI checks on 3.11. So the
only available recorder was the one that must never record. `--only` is the
way out.
"""
from pathlib import Path

import pytest

ALL = {"python_api", "pytest_cases", "c_cases", "numeric", "threads",
       "math_values"}


def test_no_only_flag_means_every_baseline():
    from check_lib import record_targets
    assert record_targets(["check.py", "--record", "tag"]) == ALL


def test_only_selects_a_subset():
    from check_lib import record_targets
    argv = ["check.py", "--record", "tag", "--only", "pytest_cases,c_cases"]
    assert record_targets(argv) == {"pytest_cases", "c_cases"}


def test_an_unknown_target_is_refused_by_name():
    from check_lib import record_targets
    argv = ["check.py", "--record", "tag", "--only", "nonsense"]
    with pytest.raises(SystemExit):
        record_targets(argv)


def test_only_without_record_is_refused():
    """A selector with nothing to select is a typo, not a no-op."""
    from check_lib import record_targets
    with pytest.raises(SystemExit):
        record_targets(["check.py", "--only", "numeric"])


# ── Refusals come before writes, not between them ──────────────────────────

def test_every_refusal_is_settled_before_the_first_baseline_is_written():
    """`refuse_short_c_inventory` used to sit INSIDE the record block, after
    `python_api.txt` and `pytest_cases.txt` had already been written. On a
    tree where the C binaries were not built, `--record TAG` therefore
    overwrote two baselines -- consuming their preserved copies, so the same
    tag could not be used again -- and only then exited 2. The first of the
    two is the file that must never be recorded on this interpreter at all.

    The same mistake this script already settles for argument faults:
    "this script does its argument checking first" (check.py:25-30).
    """
    check = Path(__file__).resolve().parents[1] / "check.py"
    source = check.read_text(encoding="utf-8")
    lines = source.splitlines()

    def line_of(needle):
        for number, line in enumerate(lines, 1):
            if needle in line:
                return number
        raise AssertionError(f"not found in check.py: {needle}")

    refusal = line_of("refuse_short_c_inventory(c_problems)")
    definition = line_of("def _write(name, text):")
    assert refusal < definition, (
        f"refuse_short_c_inventory is at line {refusal}, after _write is "
        f"defined at {definition}: a baseline can be written before the "
        f"refusal is reached")


# ── A baseline that is absent ────────────────────────────────────────────────
#
# Every check in the gate reads its baseline unconditionally and dies if the
# file is not there, which is the right behaviour: a gate that cannot find
# what it compares against has not passed, it has not run. The math-values
# check was written to print a note instead, so that the run which records
# the baseline for the first time is not blocked by its absence -- but the
# exemption has to be limited to that run, or a tree with the file missing
# reports a clean gate with the whole net silently gone.

def test_a_missing_baseline_fails_an_ordinary_run():
    from check_lib import missing_baseline_is_fatal
    assert missing_baseline_is_fatal("math_values", ["check.py"]) is True


def test_a_missing_baseline_is_allowed_only_for_the_run_recording_it():
    from check_lib import missing_baseline_is_fatal
    argv = ["check.py", "--record", "tag", "--only", "math_values"]
    assert missing_baseline_is_fatal("math_values", argv) is False


def test_a_missing_baseline_still_fails_when_another_target_is_recorded():
    """`--record --only pytest_cases` is not licence to lose math_values."""
    from check_lib import missing_baseline_is_fatal
    argv = ["check.py", "--record", "tag", "--only", "pytest_cases"]
    assert missing_baseline_is_fatal("math_values", argv) is True


def test_an_unqualified_record_run_may_find_any_baseline_absent():
    """`--record tag` writes every baseline, so any of them may be missing."""
    from check_lib import missing_baseline_is_fatal
    for name in sorted(ALL):
        assert missing_baseline_is_fatal(name, ["check.py", "--record", "t"]) is False
