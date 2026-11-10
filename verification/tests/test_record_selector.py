"""Which baselines `--record` writes.

`--record TAG` writes all five in one call, and only `.venv/bin/python` can
import BEAM here -- which is Python 3.14, where recording `python_api.txt`
would bake 3.14-only signatures into a baseline CI checks on 3.11. So the
only available recorder was the one that must never record. `--only` is the
way out.
"""
import pytest

ALL = {"python_api", "pytest_cases", "c_cases", "numeric", "threads"}


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
