"""The C runner: every binary, every failure, a thread cap.

`make test` was `for t in $(TEST_BINS); do ./$t || exit 1; done`. Three
faults in one line: it stops at the first failure, so one broken suite hides
up to 34 others; it reports nothing but the shell's exit status; and it
leaves every binary to claim all 16 cores, which at any real load is
thrashing rather than parallelism.

These tests use stand-in binaries rather than the real 35. A tooling test
that ran the suite would cost over an hour -- tonight's full run was 106
minutes -- and would be measuring the suite rather than the runner. The real
35 are covered by running `make test` itself.
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1].parent
RUNNER = ROOT / "tools" / "qa" / "run"


def _bins(tmp_path, specs):
    """specs: {name: (exit_code, seconds_to_sleep)}. Each fake prints a
    marker naming itself and the thread cap it was given."""
    d = tmp_path / "bin"
    d.mkdir(exist_ok=True)
    for name, (rc, sleep) in specs.items():
        p = d / name
        p.write_text(f"#!/bin/sh\n"
                     f"echo 'RAN {name} OMP=' \"$OMP_NUM_THREADS\"\n"
                     f"sleep {sleep}\n"
                     f"exit {rc}\n", encoding="utf-8")
        p.chmod(0o755)
    return d


def _run(bin_dir, *args):
    return subprocess.run([sys.executable, str(RUNNER), "c",
                           "--bin-dir", str(bin_dir), *args],
                          capture_output=True, text=True, encoding="utf-8",
                          cwd=ROOT)


def test_one_failing_binary_does_not_hide_the_rest(tmp_path):
    """The `|| exit 1` loop stopped here, hiding up to 34 suites."""
    d = _bins(tmp_path, {"test_aaa": (0, 0), "test_bbb": (1, 0),
                         "test_ccc": (0, 0), "test_ddd": (0, 0)})
    r = _run(d, "--jobs", "1")
    assert r.returncode != 0, r.stdout
    for name in ("test_aaa", "test_bbb", "test_ccc", "test_ddd"):
        assert name in r.stdout, f"{name} missing from the report:\n{r.stdout}"
    # The three after the failure ran, which the old loop could not do.
    assert r.stdout.count("RAN ") == 4 or "4 suites" in r.stdout


def test_the_runner_sets_a_thread_cap(tmp_path):
    """The cap is an environment variable, never an in-process call:
    src/native/core/threads.c:24-26 records that BEAM never edits global
    OpenMP state."""
    d = _bins(tmp_path, {"test_aaa": (0, 0)})
    r = _run(d, "--jobs", "1", "--threads", "2")
    assert "OMP= 2" in r.stdout, r.stdout


def test_every_suite_passing_exits_zero(tmp_path):
    d = _bins(tmp_path, {"test_aaa": (0, 0), "test_bbb": (0, 0)})
    r = _run(d, "--jobs", "2")
    assert r.returncode == 0, r.stdout


def test_a_timeout_is_not_reported_as_a_failed_assertion(tmp_path):
    """A suite killed by the wall-clock ceiling has not failed an assertion,
    and reporting it as a failure sends someone looking for a defect that is
    not there. Tonight's test_mellin_many took 1740 s under load; the
    wrapper's default ceiling is 900."""
    d = _bins(tmp_path, {"test_slow": (0, 30)})
    r = _run(d, "--jobs", "1", "--timeout", "1")
    assert r.returncode != 0
    assert "timeout" in r.stdout.lower()
    assert "1 s" in r.stdout or "1s" in r.stdout


def test_the_json_records_the_conditions_not_just_the_times(tmp_path):
    """A duration without its conditions is not a measurement: the same
    suite took 3.4 minutes on an idle box and 106 minutes at load 40."""
    d = _bins(tmp_path, {"test_aaa": (0, 0), "test_bbb": (1, 0)})
    out = tmp_path / "results.json"
    _run(d, "--jobs", "1", "--json", str(out))
    got = json.loads(out.read_text(encoding="utf-8"))
    assert got["conditions"]["threads"] == 2
    assert got["conditions"]["cores"] >= 1
    assert len(got["conditions"]["loadavg"]) == 3
    assert got["totals"]["suites"] == 2
    assert got["totals"]["failed"] == 1
    by_name = {s["name"]: s for s in got["suites"]}
    assert by_name["test_aaa"]["status"] == "pass"
    assert by_name["test_bbb"]["status"] == "fail"
    assert by_name["test_aaa"]["seconds"] >= 0


def test_a_binary_can_be_named_on_the_command_line(tmp_path):
    """Goal 3 from the other side: run one suite without running 34."""
    d = _bins(tmp_path, {"test_aaa": (0, 0), "test_bbb": (1, 0)})
    r = _run(d, "--jobs", "1", "--suite", "test_aaa")
    assert r.returncode == 0, r.stdout
    assert "test_bbb" not in r.stdout


def test_a_suite_name_that_does_not_exist_is_an_error(tmp_path):
    d = _bins(tmp_path, {"test_aaa": (0, 0)})
    r = _run(d, "--suite", "test_typo")
    assert r.returncode != 0
    assert "test_typo" in (r.stdout + r.stderr)


def test_a_wrapper_that_cannot_set_the_ceiling_is_reported_once(tmp_path):
    """CI's C-test job is checkout plus `make test` and has never used the
    wrapper. If an environment refuses the address-space ceiling, the useful
    output is one line saying so -- not 35 identical ones, and never a quiet
    run without the ceiling, which is what the wrapper exists to prevent.

    Reproduced honestly: a hard limit below the ceiling bounded asks for
    makes bounded refuse, exactly as a restricted environment would.
    """
    d = _bins(tmp_path, {"test_aaa": (0, 0), "test_bbb": (0, 0)})
    r = subprocess.run(
        ["bash", "-c", f"ulimit -v 2000000; exec {sys.executable} "
                       f"{RUNNER} c --bin-dir {d}"],
        capture_output=True, text=True, encoding="utf-8", cwd=ROOT)
    both = r.stdout + r.stderr
    assert r.returncode != 0, both
    assert "ceiling" in both.lower(), both
    assert both.count("test_aaa") <= 1, f"reported per suite, not once:\n{both}"
