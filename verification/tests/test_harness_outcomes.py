"""The C harness: outcomes including SKIP, harness-owned counters, JSON.

The old harness was 16 lines of macro and a pair of counters each test file
declared for itself. It had two outcomes, PASS and FAIL, so a test that could
not run had nowhere to say so -- one printed SKIP and then called PASS(),
which counts as verified, and one printed SKIP and returned, which counts as
nothing at all. Neither is visible in a green suite.
"""
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1].parent
BIN_DIR = ROOT / "build" / "tests" / "native"
SRC_DIR = ROOT / "tests" / "native"


def _run_json(binary, *, env_extra=None, args=()):
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "results.json"
        env = dict(os.environ)
        env.update(env_extra or {})
        r = subprocess.run([str(BIN_DIR / binary), "--json", str(out), *args],
                           capture_output=True, text=True, encoding="utf-8",
                           env=env)
        if not out.exists():
            pytest.fail(f"{binary} wrote no JSON (exit {r.returncode}):\n"
                        f"{r.stdout}\n{r.stderr}")
        return json.loads(out.read_text(encoding="utf-8"))


def test_a_skip_is_not_counted_as_a_pass():
    """Two tests print SKIP and then stop, and the green count absorbs both.

    test_the_allocator_policy_is_applied cannot check the threshold when the
    environment has overridden it, so it skips -- today by calling PASS(),
    which reports the threshold as verified when nothing was read.
    """
    r = _run_json("test_allocator_policy",
                  env_extra={"BEAM_MMAP_THRESHOLD": "131072"})
    assert r["skipped"] >= 1, r
    assert all(t["outcome"] in {"passed", "failed", "skipped"}
               for t in r["tests"]), r["tests"]
    skipped = [t for t in r["tests"] if t["outcome"] == "skipped"]
    assert any("environment" in (t["message"] or "") for t in skipped), skipped


def test_the_json_carries_the_conditions():
    r = _run_json("test_alphabet")
    for key in ("load_average", "thread_cap", "core_count"):
        assert key in r["conditions"], r["conditions"]
    assert all("seconds" in t for t in r["tests"])
    assert len(r["conditions"]["load_average"]) == 3


def test_every_test_is_accounted_for_in_one_of_the_outcomes():
    """A count that does not add up is how a test disappears quietly."""
    r = _run_json("test_alphabet")
    assert r["passed"] + r["failed"] + r["skipped"] + r["no_verdict"] \
        == len(r["tests"]) == r["total"]
    assert r["no_verdict"] == 0, [t for t in r["tests"]
                                  if t["outcome"] == "no verdict"]


def test_a_sub_case_skip_does_not_demote_a_test_that_still_verified():
    """test_row_cursor skips ONE shape when the table is too dense to seal
    and still checks the others. Reporting the whole test as skipped would
    under-report it exactly as counting it a pass over-reports the other."""
    r = _run_json("test_row_cursor")
    assert r["failed"] == 0, r["tests"]
    assert r["passed"] + r["skipped"] == r["total"]


def test_no_test_file_declares_the_counters_any_more():
    """The harness owns them. 34 files each declared their own pair, which is
    how two files could disagree about what counts as a test."""
    owned = [f.name for f in sorted(SRC_DIR.glob("test_*.c"))
             if re.search(r"static int\s+pass_count", f.read_text(encoding="utf-8"))]
    assert owned == [], owned


def test_a_binary_still_lists_the_same_way():
    """Outcomes are a reporting change; the inventory must not move."""
    out = subprocess.run([str(BIN_DIR / "test_alphabet"), "--list"],
                         capture_output=True, text=True, encoding="utf-8",
                         check=True).stdout
    assert all(n.startswith("test_") for n in out.split()), out


# ── the harness deciding, on cases built to be decided ─────────────────────
# tests/harness/selftest_outcomes.c fails on purpose, so it is not part of
# `make test`, and it lives outside tests/native so neither the suite nor the
# gate's inventory can see it.

SELFTEST = ROOT / "build" / "tests" / "harness" / "selftest_outcomes"


def _selftest():
    build = subprocess.run(["make", "selftest"], cwd=ROOT,
                           capture_output=True, text=True, encoding="utf-8")
    assert build.returncode == 0, build.stdout + build.stderr
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "results.json"
        r = subprocess.run([str(SELFTEST), "--json", str(out)],
                           capture_output=True, text=True, encoding="utf-8")
        return json.loads(out.read_text(encoding="utf-8")), r


def test_the_harness_decides_each_outcome_from_what_the_body_did():
    results, r = _selftest()
    got = {t["name"]: t["outcome"] for t in results["tests"]}
    assert got == {
        "selftest_a_pass_is_a_pass": "passed",
        "selftest_an_assertion_failure_is_a_failure": "failed",
        "selftest_a_skip_with_no_verdict_after_it_is_a_skip": "skipped",
        # A test that declines one sub-case and verifies the rest has passed.
        "selftest_a_skip_followed_by_a_verdict_is_that_verdict": "passed",
        # 74 sites across 11 files do this instead of using ASSERT_MSG.
        "selftest_a_direct_failure_count_is_a_failure": "failed",
        "selftest_a_body_that_returns_silently_has_no_verdict": "no verdict",
    }
    assert r.returncode != 0, "deliberate failures did not fail the run"


def test_a_test_that_reaches_no_verdict_fails_the_run():
    """Before the harness this was invisible: a body that returned early just
    made the pass count smaller, and nothing said which test it was."""
    results, r = _selftest()
    assert results["no_verdict"] == 1
    assert r.returncode != 0
    assert "NO VERDICT" in r.stdout


def test_every_test_gets_exactly_one_line():
    """A body that reported nothing used to leave its line unterminated, so
    the next test's name landed on the same line and both were hard to see."""
    results, r = _selftest()
    # Anchored to the recorded names, not to a prefix: the suite itself is
    # called selftest_outcomes, so its summary line shares the prefix.
    names = {t["name"] for t in results["tests"]}
    lines = [l for l in r.stdout.splitlines()
             if l.split() and l.split()[0] in names]
    assert len(lines) == len(names), r.stdout
    for line in lines:
        assert len(line.split()) >= 2, f"no verdict on the line: {line!r}"
