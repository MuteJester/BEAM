"""What every test in this suite gets: the paths, and the run's conditions.

The suite is split into tests/python/<domain>/, with shared fixtures in
tests/python/conftest.py, constants in tests/python/_helpers.py, and the
independent reference in tests/reference/. Those last two are plain modules
rather than fixtures, so their directories go on the path here.

The settings are in pytest.ini beside this file, where they travel with the
suite; this file holds only what a setting cannot express.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for d in (ROOT / "python", ROOT / "reference"):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))


def pytest_report_header(config):
    """The conditions a duration means nothing without.

    `durations = 25` in pytest.ini prints the slowest tests every run, and a
    duration without its load is not a measurement: this suite took 202 s at
    load 13 and some 800 s at load 53 on the same machine, with the same
    tests. The header carries the load, the thread cap and the core count so
    that the list below is read beside them.

    It also says when `OMP_NUM_THREADS` is unset, which is the one condition
    that turns the suite's duration into noise: BEAM applies a thread cap
    with a `num_threads` clause and never edits global OpenMP state
    (src/native/core/threads.c), so an unset cap lets every test claim every
    core, and several tests at once then thrash rather than run.

    The second line appears only when pytest read some other configuration
    than this suite's own -- a bare `pytest` at the repository root finds
    pyproject.toml first, and pyproject.toml decides nothing. Such a run
    collects the same 1,216 tests with no escalated warning and no durations
    list, and the whole point of moving the settings in here is that a run
    missing them should not look like a run that has them.
    """
    try:
        load = f"{os.getloadavg()[0]:.2f}"
    except OSError:                      # not available on every platform
        load = "unavailable"
    cap = os.environ.get("OMP_NUM_THREADS")
    cap = f"OMP_NUM_THREADS={cap}" if cap else \
        "OMP_NUM_THREADS unset (each test may claim every core)"
    lines = [f"conditions: load {load}, {cap}, {os.cpu_count()} cores"]
    if Path(str(getattr(config, "inipath", None) or "")) != ROOT / "pytest.ini":
        lines.append(
            "NOTE: tests/pytest.ini was not read, so the escalated "
            "performance warning, the declared markers and the durations "
            "list are not in force. Run `pytest tests` to get them.")
    return lines
