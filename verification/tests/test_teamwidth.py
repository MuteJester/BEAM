"""The width of the teams BEAM actually creates, measured rather than read.

Every thread check before this one was static. `check_lib.thread_count_clauses`
proves all 93 parallel regions NAME a thread count; the GIL scan proves every
GIL-releasing region RESOLVES a plan. Neither can see what the runtime granted,
and that gap is where the nesting defect lived: `beam_threads()` handed the
configured cap to a worker thread, so an 8-thread plan opened eight teams of
sixteen, while `thread counts` was OK in the very same gate run -- the clause
was present and correct and only its value on a worker was wrong.

`verification/teamwidth.c` interposes the two libgomp entry points through which
BEAM creates teams and records, per region entry, the count REQUESTED and the
width that actually RAN. Measured on one `entropy()` call, plan pinned at 8:
**949 team creations**, which is why reading 93 pragmas was never going to
answer this.

The two channels are not redundant, and the mutation that proves it is the
reason this file asserts on both. With the one-line guard removed from
`beam_threads()`:

    nesting ON  (OMP_MAX_ACTIVE_LEVELS=2)   requested 16, nested ran 16, plan 8
    nesting OFF (the default)               requested 16, nested ran  1, plan 8

So in the environment every user actually gets, the defect was INVISIBLE in the
width that ran -- libgomp declines to widen a nested region when nesting is off,
so the actual teams were a blameless 8 and 1 -- and BEAM was still emitting
`num_threads(16)` on them. Only the runtime's default policy stood between the
plan and sixteen-way oversubscription. A check on the actual width alone would
have passed on the mutant; the requested channel catches it with no special
environment at all. That is why `test_no_region_asks_for_more_than_the_plan` is
the load-bearing test here and the actual-width tests back it up.

`test_the_observer_wraps_every_team_creating_entry_point` is what keeps the
whole file honest: it reads the built extension's imports and fails if BEAM ever
grows a third way to create a team, because the observer would then be blind and
would go on reporting zero violations. And
`test_the_measurement_reached_a_nested_region` refuses to let a clean result
stand without evidence that the nested path ran at all -- a dead path and a
well-behaved one produce the same zero.
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1].parent
SHIM = ROOT / "build" / "teamwidth.so"
SHIM_SRC = ROOT / "verification" / "teamwidth.c"

requires_shim = pytest.mark.skipif(
    not SHIM.exists() or not sys.platform.startswith("linux"),
    reason="needs Linux and build/teamwidth.so; run `make teamwidth` first")


# Builds a model, asks BEAM what it planned, resets the counters, makes ONE
# call, and reports. The reset matters: without it the widest team in the
# process would include the fixture's own build, and that is not evidence
# about the call under test.
_CHILD = """
import ctypes, json, random
import BEAM
from BEAM import BeamModel

lib = ctypes.CDLL(None)
NAMES = ("outer", "nested", "widest", "requested", "unnamed", "entries", "depth")
for n in NAMES:
    f = getattr(lib, "beam_tw_" + n)
    f.restype, f.argtypes = ctypes.c_uint64, []
lib.beam_tw_reset.restype, lib.beam_tw_reset.argtypes = None, []

rng = random.Random(7)
seqs = ["TG" + "".join(rng.choice("ACGT") for _ in range(rng.randint(2, 20))) + "C"
        for _ in range(800)]
m = BeamModel(seqs, abundances=[rng.randint(1, 40) for _ in seqs],
              alphabet="nt", order=3, max_length=24)
plan = BEAM.parallel_plan()["plan"]

lib.beam_tw_reset()
h = m.entropy()
out = {n: getattr(lib, "beam_tw_" + n)() for n in NAMES}
out["plan"] = plan
out["entropy"] = h
print(json.dumps(out))
"""

_CACHE: dict[bool, dict] = {}


def _observe(nesting: bool) -> dict:
    """One child run, cached. `BEAM_AVAILABLE_CORES` pins the plan so the
    assertions do not move with the load on the machine, and OMP_NUM_THREADS is
    removed because setting it makes the policy stand aside entirely -- the plan
    would then be the environment's number and the comparison vacuous.

    Attribution rests on where the reset goes. LD_PRELOAD interposes libgomp for
    the whole process, so a counter read at exit would mix in anything else in
    the process that uses OpenMP -- a BLAS built against libgomp, for instance.
    The child resets immediately before the one call under test and reads
    immediately after, and nothing else in that window calls out to another
    library, so every region counted is BEAM's. The model is built BEFORE the
    reset for the same reason."""
    if nesting in _CACHE:
        return _CACHE[nesting]
    env = dict(os.environ, BEAM_AVAILABLE_CORES="16",
               LD_PRELOAD=str(SHIM))
    env.pop("OMP_NUM_THREADS", None)
    if nesting:
        env["OMP_MAX_ACTIVE_LEVELS"] = "2"
    else:
        env.pop("OMP_MAX_ACTIVE_LEVELS", None)
    r = subprocess.run([sys.executable, "-c", _CHILD], capture_output=True,
                       text=True, encoding="utf-8", timeout=900, env=env)
    assert r.returncode == 0, r.stderr[-3000:]
    got = json.loads(r.stdout.strip().splitlines()[-1])
    if got["plan"] < 2:
        pytest.skip(f"a plan of {got['plan']} cannot be exceeded by widening")
    _CACHE[nesting] = got
    return got


def test_the_observer_wraps_every_team_creating_entry_point():
    """The observer is only complete while BEAM creates teams through the entry
    points it interposes. A third one would leave it blind AND quiet, which is
    the worst failure a checking tool has: `nm` on the built extension is the
    only thing that can tell the difference, so the set is derived from the
    artifact and from the shim's own source, never written down here twice."""
    so = sorted(ROOT.glob("src/BEAM/_cbeam*.so"))
    if not so:
        pytest.skip("no built extension to read imports from")
    out = subprocess.run(["nm", "--undefined-only", str(so[-1])],
                         capture_output=True, text=True, encoding="utf-8")
    if out.returncode != 0:
        pytest.skip("nm unavailable")
    imported = {m.group(0) for m in
                re.finditer(r"GOMP_parallel\w*", out.stdout)}
    wrapped = set(re.findall(r"^void (GOMP_\w+)\(", SHIM_SRC.read_text(
        encoding="utf-8"), re.MULTILINE))

    missed = imported - wrapped
    assert not missed, (
        f"BEAM creates teams through {sorted(missed)}, which the observer does "
        f"not wrap: it would report zero violations while seeing none of them")


@requires_shim
def test_no_region_asks_for_more_than_the_plan():
    """The load-bearing one. A region's `num_threads(...)` is BEAM's own
    arithmetic, and it is wrong independently of whether the runtime indulges
    it: the mutant asks for 16 against a plan of 8 in the DEFAULT environment,
    where every actual team is still the correct width. Catching that needs no
    OMP_MAX_ACTIVE_LEVELS and no nesting -- only a count of what was asked."""
    got = _observe(nesting=False)
    assert got["requested"] <= got["plan"], (
        f"a region asked for {got['requested']} threads against a plan of "
        f"{got['plan']}; the runtime may have declined, which is luck, not "
        f"design: {got}")


@requires_shim
def test_no_team_runs_wider_than_the_plan():
    got = _observe(nesting=False)
    assert got["widest"] <= got["plan"], (
        f"a team ran {got['widest']} threads wide against a plan of "
        f"{got['plan']}: {got}")


@requires_shim
def test_a_nesting_environment_widens_nothing():
    """`OMP_MAX_ACTIVE_LEVELS=2` is the environment the defect needed, and a
    caller can set it: BEAM cannot pin it back without `omp_set_max_active_levels`,
    process-global state that two calls on two threads would fight over. So the
    guarantee has to hold under it rather than forbid it."""
    got = _observe(nesting=True)
    assert got["nested"] <= got["plan"] and got["widest"] <= got["plan"], (
        f"nesting widened a team past the plan: {got}")
    assert got["requested"] <= got["plan"], (
        f"nesting made a region ask past the plan: {got}")


@requires_shim
def test_every_region_that_ran_named_a_thread_count():
    """The runtime counterpart of `check_lib.thread_count_clauses`. That check
    reads the sources and so covers regions whether or not they execute; this
    one covers regions that executed whether or not the source scan understood
    them. A zero here means no region reached libgomp with `nthreads == 0`,
    which is what a missing clause looks like from the outside."""
    got = _observe(nesting=False)
    assert got["unnamed"] == 0, (
        f"{got['unnamed']} of {got['entries']} regions ran without naming a "
        f"thread count: {got}")


@requires_shim
def test_the_measurement_reached_a_nested_region():
    """Without this, every other assertion here could be passing because the
    nested path never ran. A dead path and a well-behaved one both report
    nothing, and the parallelism work has already been bitten once by a check
    whose claim outran its evidence."""
    got = _observe(nesting=True)
    assert got["entries"] > 0, f"the observer saw no teams at all: {got}"
    assert got["depth"] >= 2, (
        f"no nested region was entered, so the nesting assertions prove "
        f"nothing: {got}")
