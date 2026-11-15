"""The checks this repository owns are of no use unless CI runs them.

Two of them were not run by any workflow, and both had the same shape as the
verification gate before `845a4f2`: a check nobody runs does not report a
regression late, it reports nothing.

  * `verification/tests` -- the suite guarding the gate itself, including two
    hardcoded tripwires (the `ALL` set in `test_record_selector`, the Makefile
    target spelling in `test_wrap_link_guard`) that only pay off if something
    runs them.
  * More than one thread environment. Two real defects in the parallelism work
    appeared ONLY under a non-default `OMP_NUM_THREADS`: an assertion whose two
    outcomes collided on the same number, and a policy clause reading
    `omp_get_num_procs()` where it needed `omp_get_max_threads()`. The C runner
    pins `OMP_NUM_THREADS=2` for every suite, so a single run of it cannot see
    that class at all.

WHAT THESE TESTS PROVE, AND WHAT THEY DO NOT. They read the workflow text and
prove a command is PRESENT in it, outside a comment. They do not prove the job
is reached, that its `if:` lets it run, or that it passed -- the same
limitation the thread-plan wiring check turned out to have, and worth stating
here rather than discovering later. What they are is a tripwire: delete the
job and a test fails, which is the property that was missing.

Deliberately text-based. Parsing the YAML properly would mean adding pyyaml to
the dev dependencies for one tripwire, and these assertions are about commands
appearing in a file, which does not need a parser.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

WORKFLOWS = Path(__file__).resolve().parents[2] / ".github" / "workflows"

# The C runner's own default, which every CI run of the C suite has used. A
# second environment has to differ from THIS, not merely exist.
RUNNER_DEFAULT_THREADS = 2


def workflow_lines() -> list[tuple[str, str]]:
    """Every (file, line) in the workflows, with comment-only lines dropped.

    A `#` inside a quoted string is not a comment, but no workflow here has
    one, and the alternative is a YAML parser. Lines whose first non-space
    character is `#` are the ones that matter: commenting a job out is how a
    check stops running without anyone deleting it.
    """
    out = []
    for path in sorted(WORKFLOWS.glob("*.yml")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("#"):
                continue
            out.append((path.name, line))
    return out


def test_some_workflow_runs_the_tooling_suite():
    """verification/tests runs in CI, not only by hand."""
    hits = [(f, l) for (f, l) in workflow_lines()
            if "verification/tests" in l and "pytest" in l]
    assert hits, (
        "No workflow runs `pytest verification/tests`. These are the tests "
        "that guard the gate; running them by hand means they guard it only "
        "on the days someone remembers."
    )


def test_the_c_suite_runs_at_more_than_one_thread_count():
    """Some workflow runs the C suite at a thread count other than the default."""
    counts = set()
    for (_f, line) in workflow_lines():
        for m in re.finditer(r"--threads[= ]+(\d+)", line):
            counts.add(int(m.group(1)))
    extra = counts - {RUNNER_DEFAULT_THREADS}
    assert extra, (
        f"Every CI run of the C suite uses the runner default of "
        f"{RUNNER_DEFAULT_THREADS} threads (found: {sorted(counts) or 'none'}). "
        "Two real defects in this work were visible only at a different "
        "thread count, so one environment cannot find that class."
    )


def test_a_thread_count_of_one_is_among_them():
    """One thread is the environment that catches 'assumes a team'."""
    counts = {int(m.group(1))
              for (_f, line) in workflow_lines()
              for m in re.finditer(r"--threads[= ]+(\d+)", line)}
    assert 1 in counts, (
        f"No workflow runs the C suite single-threaded (found: {sorted(counts)}). "
        "A team of one is the case where a reduction, a barrier or a "
        "per-thread buffer can be skipped entirely."
    )


def test_an_odd_thread_count_is_among_them():
    """An odd count catches 'assumes the work divides evenly'."""
    counts = {int(m.group(1))
              for (_f, line) in workflow_lines()
              for m in re.finditer(r"--threads[= ]+(\d+)", line)}
    odd = {c for c in counts if c > 1 and c % 2 == 1}
    assert odd, (
        f"No workflow runs the C suite at an odd thread count above one "
        f"(found: {sorted(counts)}). Powers of two hide off-by-one chunking: "
        "every count CI has ever used divides the chunk counts evenly."
    )


def test_the_python_suite_sees_a_pinned_thread_environment():
    """Some workflow sets OMP_NUM_THREADS for Python tests."""
    hits = [(f, l) for (f, l) in workflow_lines() if "OMP_NUM_THREADS" in l]
    assert hits, (
        "No workflow sets OMP_NUM_THREADS. The Python legs therefore all run "
        "at whatever the runner's core count implies, which is one "
        "environment chosen by the hardware rather than by this repository."
    )


@pytest.mark.parametrize("command", [
    "verification/check.py",        # the gate
    "make -j",                      # the C suite
])
def test_the_checks_that_already_ran_are_still_wired(command):
    """A regression guard for the jobs that were already there.

    Without this, a later edit could satisfy the tests above while quietly
    dropping the checks CI did run.
    """
    assert any(command in l for (_f, l) in workflow_lines()), \
        f"No workflow runs `{command}` any more."
