"""The pytest configuration must travel with the suite.

CI runs the Python half from a COPY of `tests/`, against the installed
package:

    cp -r tests "$RUNNER_TEMP/tests" && cd "$RUNNER_TEMP" && pytest tests

pytest looks for its configuration upward from its arguments, so from that
copy `pyproject.toml` is never in sight -- and that is where every setting
used to live. The escalated warning, the markers and the durations list were
therefore in force on no CI leg at all, which is the one place they matter
most. They now live in `tests/pytest.ini`, and these tests pin both halves of
that: the copy finds the configuration, and the repository root and the copy
agree about what they are running.

These tests live here and not under `tests/` because they read
`pyproject.toml` and compare two invocations of the suite -- both outside the
suite, which must stay self-contained.
"""
import os
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import check_lib

ROOT = Path(__file__).resolve().parents[2]
SUITE = ROOT / "tests"
NOISE = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache")
# Two threads, not sixteen: these tests start pytest in a subprocess, and a
# nested run that claims every core competes with the run that started it.
ENV = {**os.environ, "OMP_NUM_THREADS": "2"}

# One file is enough to ask which configuration pytest found, and asking with
# the whole suite would print 1,216 collected items to read three lines of
# header out of.
ONE_FILE = "tests/suites/model/test_warnings.py"

# One test, chosen because it asserts a subclass relation and so costs
# nothing but the import: these two tests need a RUN, not a collection.
FAST_TEST = "tests/suites/model/test_warnings.py::test_the_warning_is_a_user_warning"

PROBE = '''import warnings

import BEAM


def test_the_probe_emits_the_performance_warning():
    warnings.warn("probe", BEAM.BeamPerformanceWarning)
'''

MARKER_PROBE = '''import pytest


@pytest.mark.slow
def test_a_declared_marker_is_accepted():
    pass


@pytest.mark.slwo
def test_a_marker_that_is_a_typo_is_refused():
    pass
'''


def _copy_of_the_suite(tmp_path):
    """`tests/` alone, where nothing above it can be read."""
    shutil.copytree(SUITE, tmp_path / "tests", ignore=NOISE)
    return tmp_path


def _pytest(args, cwd):
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", *args],
        capture_output=True, text=True, encoding="utf-8", cwd=cwd, env=ENV)


def test_the_configuration_is_found_from_a_copy_of_the_suite_alone(tmp_path):
    out = _pytest([ONE_FILE, "--collect-only"],
                  cwd=_copy_of_the_suite(tmp_path)).stdout
    assert "configfile: pytest.ini" in out, out[:1500]


def test_the_performance_warning_is_an_error_by_configuration(tmp_path):
    """Without a `-W` flag on the command line, which CI does not pass."""
    root = _copy_of_the_suite(tmp_path)
    probe = root / "tests" / "suites" / "test_probe_performance_warning.py"
    probe.write_text(PROBE, encoding="utf-8")
    run = _pytest([str(probe.relative_to(root)), "-q"], cwd=root)
    assert run.returncode != 0, run.stdout[-1500:]
    assert "BeamPerformanceWarning" in run.stdout, run.stdout[-1500:]


def test_the_repository_root_and_the_copy_collect_the_same_ids(tmp_path):
    """The two invocations this project actually uses. They disagreed: with
    the configuration in `pyproject.toml` the root numbered every id from the
    repository and the copy numbered it from `tests/`, so the same test had
    two names depending on where it was run."""
    here = _pytest(["tests", "--collect-only", "-q"], cwd=ROOT)
    there = _pytest(["tests", "--collect-only", "-q"],
                    cwd=_copy_of_the_suite(tmp_path))
    assert check_lib.collect_ids(here.stdout) == check_lib.collect_ids(there.stdout)


def test_the_suite_declares_the_markers_it_selects_by():
    out = _pytest(["tests", "--markers"], cwd=ROOT).stdout
    for marker in ("@pytest.mark.slow", "@pytest.mark.oracle"):
        assert marker in out, out[:1500]


def test_the_run_header_carries_the_conditions_a_duration_means_nothing_without():
    """`durations = 25` prints the slowest tests every run. The same suite
    took 202 s at load 13 and about 800 s at load 53 -- same tests, same box
    -- so the list is only readable beside the load, the thread cap and the
    core count."""
    out = _pytest([ONE_FILE, "--collect-only"], cwd=ROOT).stdout
    assert "conditions:" in out, out[:1500]
    for field in ("load", "OMP_NUM_THREADS", "cores"):
        assert field in out, out[:1500]


def test_pyproject_says_where_the_suite_is_and_nothing_else():
    """Two configuration files cannot diverge if only one of them decides
    anything. `testpaths` stays because a bare `pytest` at the repository
    root would otherwise sweep .venv/ and docs/ as well."""
    cfg = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    ini = cfg.get("tool", {}).get("pytest", {}).get("ini_options", {})
    assert set(ini) <= {"testpaths"}, sorted(ini)


def test_the_gate_numbers_its_ids_from_the_repository_root():
    """pytest numbers ids from the rootdir, and the rootdir follows the
    configuration file: `tests/pytest.ini` makes it `tests/`, which would
    strip the `tests/` prefix from all 1,216 recorded ids at once. The
    baseline compares whole ids, so the gate pins the rootdir instead."""
    out = subprocess.run(check_lib.python_collect_argv(ROOT), cwd=ROOT,
                         capture_output=True, text=True,
                         encoding="utf-8", env=ENV).stdout
    ids = check_lib.collect_ids(out)
    assert ids, out[-1500:]
    assert all(i.startswith("tests/") for i in ids), ids[:5]


def test_a_run_that_missed_the_suite_configuration_says_so():
    """`testpaths` in pyproject.toml means a bare `pytest` at the repository
    root still collects this suite and only this suite -- but it reads
    pyproject.toml, which decides nothing, so that run has no escalated
    warning and no durations list. It says so, instead of looking exactly
    like a configured one."""
    bare = _pytest(["--collect-only"], cwd=ROOT).stdout
    assert "tests/pytest.ini was not read" in bare, bare[:1500]
    configured = _pytest([ONE_FILE, "--collect-only"], cwd=ROOT).stdout
    assert "was not read" not in configured, configured[:1500]


def test_the_slowest_tests_are_listed_without_being_asked_for():
    out = _pytest([FAST_TEST, "-q"], cwd=ROOT).stdout
    assert "slowest 25 durations" in out, out[-1500:]


def test_no_line_of_the_configuration_is_a_line_this_pytest_ignores():
    """`--strict-config` fails a run only when it is on the command line;
    from `addopts` it warns, and a warning in a 1,216-test run is not read.
    So a key this pytest does not have would sit in the file doing nothing,
    which is what `durations = 25` did: there is no such ini option in
    pytest 9, only the `--durations` flag."""
    run = _pytest([FAST_TEST, "-q"], cwd=ROOT)
    assert "Unknown config option" not in run.stdout + run.stderr, \
        (run.stdout + run.stderr)[-1500:]



def test_a_marker_the_configuration_does_not_declare_refuses_the_run(tmp_path):
    """`--strict-markers` in `addopts` does nothing at all: pytest 9 reads
    the strictness from an ini KEY (_pytest/mark/structures.py:588), so the
    flag sat in the file while an undeclared marker stayed a line in the
    warnings summary of a 1,216-test run -- which is where warnings go to be
    unread. Declaring markers is only worth anything if a typo is refused."""
    root = _copy_of_the_suite(tmp_path)
    probe = root / "tests" / "suites" / "test_marker_probe.py"
    probe.write_text(MARKER_PROBE, encoding="utf-8")
    run = _pytest([str(probe.relative_to(root)), "-q"], cwd=root)
    assert run.returncode != 0, run.stdout[-1500:]
    assert "'slwo' not found in `markers`" in run.stdout, run.stdout[-1500:]


def test_a_setting_this_pytest_does_not_have_refuses_the_run(tmp_path):
    """The same hole, on the other key: from `addopts`, `--strict-config`
    only warns. `durations = 25` -- an ini option pytest 9 does not have --
    therefore sat in this file doing nothing, and was found by reading the
    output of a one-test run rather than by any guard."""
    root = _copy_of_the_suite(tmp_path)
    ini = root / "tests" / "pytest.ini"
    ini.write_text(ini.read_text(encoding="utf-8") + "\nno_such_option = 1\n",
                   encoding="utf-8")
    run = _pytest([FAST_TEST, "-q"], cwd=root)
    assert run.returncode != 0, run.stdout[-1500:]
    assert "Unknown config option: no_such_option" in run.stdout + run.stderr, \
        (run.stdout + run.stderr)[-1500:]


def test_a_bare_sibling_import_fails_loudly_rather_than_resolving(tmp_path):
    """The suite's helper modules used to resolve by accident: with no
    `__init__.py` under the suites, pytest's `prepend` import mode put
    each test file's OWN directory on `sys.path` as it collected it, so a
    module sitting beside a test was importable by its bare name and nothing
    said so. `--import-mode=importlib` inserts nothing, which is what makes a
    stale import an error instead of a quiet success.

    The probe imports a module that EXISTS, beside the test that imports it.
    Under the old mode that is a pass."""
    root = _copy_of_the_suite(tmp_path)
    here = root / "tests" / "suites"
    (here / "_probe_sibling.py").write_text("VALUE = 1\n", encoding="utf-8")
    probe = here / "test_probe_sibling_import.py"
    probe.write_text("from _probe_sibling import VALUE\n\n\n"
                     "def test_the_sibling_was_importable():\n"
                     "    assert VALUE == 1\n", encoding="utf-8")
    run = _pytest([str(probe.relative_to(root)), "-q"], cwd=root)
    assert run.returncode != 0, run.stdout[-1500:]
    assert "ModuleNotFoundError" in run.stdout + run.stderr, \
        (run.stdout + run.stderr)[-1500:]
