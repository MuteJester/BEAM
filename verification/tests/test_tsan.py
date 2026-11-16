"""The race detector, and the three settings without which it reports noise.

`make tsan` builds the suites instrumented; nothing ran them until now, and
running them wrong is worse than not running them. Three settings are
mandatory and none is obvious:

  * Archer must be loaded as an OMPT tool. libomp carries no sanitiser
    annotations, so without it TSan sees no happens-before edge between any
    two OpenMP threads: every implicit barrier is invisible and every region
    whose result somebody later reads reports as a race. Measured on
    test_distance alone at 8 threads: 783 reports with Archer off, 0 with it
    on. Under GCC/libgomp the same suite gave 1186. A thousand false reports
    is worse than no tool.

  * `ignore_noninstrumented_modules=1`, which Archer itself asks for on
    startup. Without it the uninstrumented runtime's own internals report.

  * No address-space ceiling. TSan maps shadow memory across the address
    space, so under `ulimit -v` it refuses outright: "Program is run with
    limited virtual address space, which wouldn't work with ThreadSanitizer
    ... setrlimit() failed 22". `verification/bounded` sets exactly that
    ceiling, so the instrumented run cannot go through it, and TSan's own
    `max_allocation_size_mb` takes over the job the ceiling was there to do.

The last test here is the one that matters: it proves the detector has teeth
by running a real suite with Archer off and watching the reports appear. A
clean run is only evidence if a dirty one is reachable.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1].parent
RUNNER = ROOT / "tools" / "qa" / "run"
SUITE_DIR = ROOT / "tests" / "native"
TSAN_BINS = ROOT / "build-tsan" / "tests" / "native"


def _fake(tmp_path, name="test_env", exit_code=0):
    """A stand-in suite that prints the two environment variables and the
    address-space limit it was actually given. Printing what the child sees
    is the only honest check: a test that read the runner's source would pass
    on a runner that built the environment and then dropped it."""
    d = tmp_path / "bin"
    d.mkdir(exist_ok=True)
    p = d / name
    p.write_text("#!/bin/bash\n"
                 'echo "TOOL=$OMP_TOOL_LIBRARIES"\n'
                 'echo "TSAN_OPTIONS=$TSAN_OPTIONS"\n'
                 'echo "ULIMIT_V=$(ulimit -v)"\n'
                 f"exit {exit_code}\n", encoding="utf-8")
    p.chmod(0o755)
    return d


def _archer(tmp_path):
    """A stand-in Archer. The runner only has to find and pass it; loading it
    is libomp's business, and a real one would make these tests depend on a
    clang install."""
    lib = tmp_path / "libarcher.so"
    lib.write_bytes(b"")
    return lib


def _run(bin_dir, *args):
    return subprocess.run([sys.executable, str(RUNNER), "c",
                           "--bin-dir", str(bin_dir), *args],
                          capture_output=True, text=True, encoding="utf-8",
                          cwd=ROOT)


def test_the_tsan_mode_loads_archer(tmp_path):
    """Without this the run is 783 false reports on one suite."""
    d = _fake(tmp_path)
    r = _run(d, "--tsan", "--archer", str(_archer(tmp_path)), "--jobs", "1")
    assert f"TOOL={_archer(tmp_path)}" in r.stdout, r.stdout


def test_the_tsan_mode_silences_the_uninstrumented_runtime(tmp_path):
    """Archer prints this request itself on startup; the runner should not
    make someone read the startup banner to learn it."""
    d = _fake(tmp_path)
    r = _run(d, "--tsan", "--archer", str(_archer(tmp_path)), "--jobs", "1")
    line = next(x for x in r.stdout.splitlines()
                if x.startswith("TSAN_OPTIONS="))
    assert "ignore_noninstrumented_modules=1" in line, line


def test_the_tsan_mode_does_not_impose_an_address_space_ceiling(tmp_path):
    """TSan refuses to start under `ulimit -v`, so the wrapper that every
    other run goes through has to be out of this one's way."""
    d = _fake(tmp_path)
    r = _run(d, "--tsan", "--archer", str(_archer(tmp_path)), "--jobs", "1")
    assert "ULIMIT_V=unlimited" in r.stdout, r.stdout


def test_the_tsan_mode_keeps_a_ceiling_on_the_allocator_instead(tmp_path):
    """The ceiling exists because BEAM can be asked for tables of hundreds of
    GB and Linux grants them lazily, so the failure lands on whatever the OOM
    killer picks. Dropping `ulimit -v` without putting anything in its place
    would hand that hazard back."""
    d = _fake(tmp_path)
    r = _run(d, "--tsan", "--archer", str(_archer(tmp_path)),
             "--jobs", "1", "--mem-gb", "7")
    line = next(x for x in r.stdout.splitlines()
                if x.startswith("TSAN_OPTIONS="))
    assert "max_allocation_size_mb=7168" in line, line
    assert "allocator_may_return_null=1" in line, line


def test_the_normal_mode_still_imposes_the_ceiling(tmp_path):
    """The complement of the two above: the escape is for the instrumented
    run only, and a refactor that widened it would be invisible otherwise."""
    d = _fake(tmp_path)
    r = _run(d, "--jobs", "1", "--mem-gb", "7")
    assert "ULIMIT_V=7340032" in r.stdout, r.stdout


def test_the_tsan_mode_refuses_without_archer(tmp_path):
    """A missing Archer must stop the run, not start one whose every report
    is a barrier artifact. The failure mode this guards is a readable log
    full of races that are not there -- which is how a week gets spent."""
    d = _fake(tmp_path)
    r = _run(d, "--tsan", "--archer", str(tmp_path / "absent.so"),
             "--jobs", "1")
    assert r.returncode == 2, f"exit {r.returncode}\n{r.stdout}{r.stderr}"
    # A phrase the runner prints, not one argparse would: "unrecognized
    # arguments: --archer /tmp/.../absent.so" also contains "archer", so
    # matching that word would make this test pass before the mode exists.
    assert "no Archer at" in (r.stdout + r.stderr), r.stdout + r.stderr
    assert "TOOL=" not in r.stdout, "the suite ran anyway"


def test_a_race_report_is_not_reported_as_a_failed_assertion(tmp_path):
    """TSan exits 66 when it reported something. An assertion failure exits
    1. Calling both "failed" sends someone hunting a broken test case when
    the test case is fine and the library is not."""
    d = _fake(tmp_path, exit_code=66)
    r = _run(d, "--tsan", "--archer", str(_archer(tmp_path)), "--jobs", "1")
    assert r.returncode != 0
    assert "RACES" in r.stdout, r.stdout
    assert "1 with races" in r.stdout or "races: 1" in r.stdout, r.stdout


def test_the_json_counts_the_reports_per_suite(tmp_path):
    """A count per suite is what makes a second run comparable. The headline
    number of this work is 783 -> 0 on one suite; that only means something
    because both were counted the same way."""
    d = _fake(tmp_path, exit_code=66)
    out = tmp_path / "results.json"
    r = _run(d, "--tsan", "--archer", str(_archer(tmp_path)), "--jobs", "1",
             "--json", str(out))
    assert out.exists(), r.stdout + r.stderr
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["conditions"]["tsan"] is True
    assert "reports" in data["suites"][0], data["suites"][0]


def test_the_instrumented_build_enables_avx():
    """-mavx is not an optimisation. src/native/math/vtrig.c passes a 32-byte
    vector by value to a static helper, which clang reports as an ABI change
    when AVX is off. While target_clones was in play that was a hard error --
    the default clone cannot satisfy it, and __attribute__((always_inline))
    does not help, the check being in the frontend before inlining -- and it
    is why the flag went in. Now that BEAM_TSAN turns the clones off it is a
    -Wpsabi warning instead, so the flag is kept for two smaller reasons: the
    instrumented build makes that call with the same vector ABI as the
    shipping build, and two warnings per build stay out of a log where a new
    warning should stand out."""
    makefile = re.sub(r"\\\n\s*", " ",
                      (ROOT / "Makefile").read_text(encoding="utf-8"))
    # Exactly one make recipe: the target line plus the tab-indented lines
    # under it. An earlier version ran to the next line starting with a word
    # character, which swallowed whatever target came next -- adding the
    # teamwidth target between `tsan` and `clean` grew the match from 10 lines
    # to 26, so this test would have passed on a Makefile where -mavx had
    # drifted into a NEIGHBOURING recipe and out of this one.
    recipe = re.search(r"^tsan:[^\n]*\n(?:\t[^\n]*\n)*", makefile, re.MULTILINE)
    assert recipe, "no tsan target in the Makefile"
    assert re.search(r"SANFLAGS=\"[^\"]*-mavx", recipe.group(0)), \
        "the tsan recipe does not pass -mavx; vtrig.c will not compile"


def _makefile() -> str:
    return re.sub(r"\\\n\s*", " ",
                  (ROOT / "Makefile").read_text(encoding="utf-8"))


def _tsan_excluded() -> set[str]:
    line = re.search(r"TSAN_EXCLUDE\s*:?=([^\n]*)", _makefile())
    assert line, "no TSAN_EXCLUDE in the Makefile"
    return set(line.group(1).split())


def test_multiversioned_code_is_single_versioned_under_the_sanitiser():
    """`target_clones` makes the compiler emit an IFUNC resolver, and the
    dynamic loader calls it during startup relocation -- before the TSan
    runtime has initialised. Clang instruments the resolver too, so
    __tsan_func_entry runs against uninitialised state and the process dies
    inside ld.so, before main, printing nothing and exiting 139.

    Nine of the 35 suites died exactly that way, and the empty log is what
    makes this worth a tripwire: there is no message to search for, no stack
    without a debugger, and the next person to add a multiversioned function
    would meet it again from scratch. The guard belongs on the attribute, so
    it travels with the thing that causes it."""
    offenders = []
    for source in sorted((ROOT / "src").rglob("*.c")):
        lines = source.read_text(encoding="utf-8").splitlines()
        for n, line in enumerate(lines):
            if "target_clones" not in line:
                continue
            guard = next((lines[m] for m in range(n - 1, max(-1, n - 8), -1)
                          if lines[m].lstrip().startswith("#if")), "")
            if "BEAM_TSAN" not in guard:
                offenders.append(f"{source.relative_to(ROOT)}:{n + 1} "
                                 f"guarded by {guard.strip() or '(nothing)'}")
    assert not offenders, \
        "these will crash the instrumented build inside ld.so:\n  " + \
        "\n  ".join(offenders)


def test_suites_that_measure_the_allocator_are_not_instrumented():
    """TSan replaces the allocator and maps shadow memory several times the
    heap, so an assertion about resident memory or about glibc's mmap
    threshold measures the sanitiser rather than BEAM. Both allocator suites
    do precisely that -- one failed at "resident memory is many times the
    model's own size", the other at "the allocator policy did not reach this
    binary" -- and a suite that cannot pass under instrumentation has to be
    excluded rather than left to fail, or the run grows a failure everyone
    learns to skip over.

    Derived from the sources, not listed here, so a new suite of this kind
    forces its own exclusion."""
    excluded = _tsan_excluded()
    for source in sorted(SUITE_DIR.glob("test_*.c")):
        text = source.read_text(encoding="utf-8")
        if re.search(r"statm|VmRSS|mallopt|M_MMAP_THRESHOLD"
                     r"|beam_applied_mmap_threshold", text):
            assert source.stem in excluded, \
                f"{source.name} measures the allocator but is instrumented"


@pytest.mark.skipif(not TSAN_BINS.is_dir(),
                    reason="no build-tsan; run `make tsan` first")
def test_every_instrumented_suite_gets_as_far_as_its_own_main():
    """The cheapest possible check, and the one that caught the IFUNC crash:
    `--list` prints the suite's test names and exits, so it costs
    milliseconds while still going through the dynamic relocation where a
    load-time crash happens. A suite that cannot reach main reports no
    races -- which reads, in a summary, exactly like a suite that has none.
    """
    built = sorted(p for p in TSAN_BINS.glob("test_*") if p.is_file())
    assert built, "build-tsan exists but holds no suites"
    dead = []
    for binary in built:
        r = subprocess.run([str(binary), "--list"], capture_output=True,
                           text=True, encoding="utf-8", errors="replace",
                           timeout=120)
        if r.returncode != 0:
            dead.append(f"{binary.name} exit {r.returncode} "
                        f"({len(r.stdout)} bytes of output)")
    assert not dead, "instrumented suites that never reach main:\n  " + \
                     "\n  ".join(dead)


@pytest.mark.skipif(not TSAN_BINS.is_dir(),
                    reason="no build-tsan; run `make tsan` first")
def test_archer_is_what_makes_a_clean_run_mean_anything():
    """The teeth of every other test here. A clean instrumented run proves
    nothing unless a dirty one is reachable from the same binary, and the
    only difference between the two is whether Archer was loaded. An idle
    machine and a dead code path produce identical output; so do a race-free
    library and a detector that was never attached."""
    suite = next((p for p in sorted(TSAN_BINS.glob("test_*"))
                  if p.name == "test_distance"), None)
    if suite is None:
        pytest.skip("test_distance not among the instrumented suites")

    def reports(archer: bool) -> int:
        env = {"OMP_NUM_THREADS": "8", "PATH": "/usr/bin:/bin",
               "TSAN_OPTIONS": "ignore_noninstrumented_modules=1"}
        if archer:
            found = sorted(Path("/usr/lib").glob("llvm-*/lib/libarcher.so"))
            if not found:
                pytest.skip("no libarcher.so installed")
            env["OMP_TOOL_LIBRARIES"] = str(found[-1])
        r = subprocess.run([str(suite)], capture_output=True, text=True,
                           encoding="utf-8", errors="replace", env=env,
                           timeout=900)
        return len(re.findall(r"WARNING: ThreadSanitizer", r.stdout + r.stderr))

    without = reports(archer=False)
    assert without > 0, \
        "with Archer off this suite reported nothing, so a clean run with " \
        "Archer on is not evidence of anything"
    assert reports(archer=True) == 0, \
        "the instrumented suite reports a race with Archer loaded"
