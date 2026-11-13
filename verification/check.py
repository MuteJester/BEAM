#!/usr/bin/env python3
"""Compare the current build against the Phase 0 baseline.

Every quantity is compared as a hex float, so this fails on a one-bit change
that a tolerance-based assertion would pass. That is the point: the phases
that split numerical code into new translation units can alter inlining, and
inlining can alter the last bit. A green test suite does not prove that did
not happen; this does.

Exit status is 0 only if the public API, the C symbols, the test inventory and
every recorded number are unchanged. All four are actually read: an earlier
version of this file claimed the API and inventory checks in its docstring and
implemented neither, which a reviewer caught.
"""
import hashlib, inspect, json, re, subprocess, sys
from pathlib import Path

import numpy as np
import BEAM

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_lib import (c_inventory, collect_ids, compare,
                       inventory_difference, missing_baseline_is_fatal,
                       python_collect_argv, python_inventory,
                       record_targets, refuse_short_c_inventory)
from math_values import capture as math_capture, differences as math_differences

# Argument faults are settled BEFORE the checks run: this script does its
# work at module level, so a refusal placed after it costs a full gate run
# first -- which is how this one was first written. `--only` without
# `--record` selects nothing, so it is a typo rather than a no-op.
if "--only" in sys.argv and "--record" not in sys.argv:
    record_targets(sys.argv)

B = Path(__file__).resolve().parent / "baseline"
fail = []


def public_text_names(header_text, syms):
    """Symbols a public header actually DECLARES.

    A plain `s in header_text` was a substring test over the raw header, so
    `beam_mix` matched inside `beam_mixture_support_union` and `jet_sweep`
    matched a prose comment -- two names counted as public API that no header
    declares. Requiring the name to be followed by `(` at a word boundary is
    still cheap and still text-based, and it costs those two their place.
    """
    return {s for s in syms
            if re.search(r"\b" + re.escape(s) + r"\s*\(", header_text)}


def check(name, got, want):
    # Compare name lists as MULTISETS. A name that appears in two files must
    # not collapse into one entry, or deleting either copy is invisible --
    # four ids are in that position. Ordering still does not matter: GNU sort
    # and Python order underscores differently depending on locale, so
    # comparing sequences would report a difference between two identical
    # inventories.
    same = compare(got, want) if isinstance(got, list) and isinstance(want, list) \
        else got == want
    if isinstance(got, list) and isinstance(want, list):
        got, want = sorted(got), sorted(want)
    if same:
        print(f"  OK    {name}")
    else:
        fail.append(name)
        print(f"  FAIL  {name}")
        if isinstance(want, list) and isinstance(got, list):
            # Counted, not membership-tested: a name that survives in one
            # file and is deleted from another is still "in" got, and the
            # old form printed nothing for exactly the case the multiset
            # comparison exists to catch.
            miss, extra = inventory_difference(got, want)
            if miss:  print(f"          missing : {miss[:6]}")
            if extra: print(f"          added   : {extra[:6]}")


# ── The binary under test ──────────────────────────────────────────────────
#
# Everything below reads `BEAM._cbeam`, which is a DIFFERENT artifact from
# build/libbeam.a. `make` builds the static library and the C tests; the
# extension module is built separately by setup.py, and nothing rebuilds it.
# So the C suite can pass against fresh code while every Python check here
# validates a stale binary -- silently, with no warning, and with the numeric
# checks below reporting that nothing changed because in that binary nothing
# did.
#
# Not hypothetical. A six-hour-old .so reported a 15.01x memory ratio for a
# change that measured 1.36x through the C benchmark, from the same source
# tree: the fix was real and reached nobody. Checked first, because a stale
# binary makes every result after it meaningless rather than merely wrong.
ROOT = Path(__file__).resolve().parents[1]
_so = Path(BEAM._cbeam.__file__)
_newest, _newest_src = 0.0, None
for _pat in ("src/native/**/*.c", "src/native/**/*.h", "include/beam/*.h"):
    for _f in ROOT.glob(_pat):
        _t = _f.stat().st_mtime
        if _t > _newest:
            _newest, _newest_src = _t, _f
if _newest_src is not None and _so.stat().st_mtime < _newest:
    fail.append("extension freshness")
    print("  FAIL  extension freshness")
    print(f"          {_so.name}")
    print(f"          is older than {_newest_src.relative_to(ROOT)}")
    print("          rebuild:  python setup.py build_ext --inplace")
    print("          until then every check below tests the PREVIOUS binary")
else:
    print("  OK    extension freshness")


# ── Public surfaces ────────────────────────────────────────────────────────
check("_cbeam callables",
      sorted(n for n in dir(BEAM._cbeam) if not n.startswith("__")),
      (B / "cbeam_methods.txt").read_text().split())

lib = ROOT / "build" / "libbeam.a"
if lib.exists():
    out = subprocess.run(["nm", "-g", "--defined-only", str(lib)],
                         capture_output=True, text=True).stdout
    syms = {l.split()[2] for l in out.splitlines()
            if len(l.split()) == 3 and l.split()[1] == "T"}

    # The contract is the PUBLIC surface: what a consumer can reach through
    # include/beam/. Splitting a file can give an internal helper external
    # linkage, which shows up in nm but is not API -- it is declared only
    # under src/native/internal/ and no public header mentions it. Those are
    # reported, not failed. A public symbol going missing still fails.
    public_text = "\n".join(h.read_text() for h in (ROOT / "include" / "beam").glob("*.h"))
    baseline = set((B / "c_symbols.txt").read_text().split())
    #
    # WHICH symbols were public is recorded, not recomputed from the current
    # headers. Recomputing makes deleting a declaration self-approving: the
    # symbol leaves both sides of the comparison and the removal disappears
    # with it. That was tried, and it silently passed a public symbol being
    # deleted from include/beam/ and made static.
    #
    public_baseline = set((B / "c_public_symbols.txt").read_text().split())
    check("C public symbols", sorted(public_text_names(public_text, syms)),
          sorted(public_baseline))
    internal_new = sorted(syms - baseline)
    if internal_new:
        print(f"  note  {len(internal_new)} internal symbol(s) newly external: "
              f"{', '.join(internal_new)}")
    #
    # A symbol leaving the export table is only a failure if it was PUBLIC.
    #
    # This used to fail on any baseline symbol that disappeared, and the
    # baseline pins 73 internal ones -- 13 of which are used in a single .c
    # file and could be `static` today. Making one of them static, which is
    # the encapsulation this check exists to encourage, produced
    # `FAIL symbols removed` on a build that compiled and passed every test.
    # The gate was pointed at the wrong direction: it failed the improvement
    # and merely noted the regression.
    #
    gone_public = sorted(public_baseline - syms)
    if gone_public:
        fail.append("public symbols removed")
        print(f"  FAIL  public symbols removed: {', '.join(gone_public)}")
    gone_internal = sorted((baseline - public_baseline) - syms)
    if gone_internal:
        print(f"  note  {len(gone_internal)} internal symbol(s) no longer "
              f"exported: {', '.join(gone_internal)}")
else:
    print("  SKIP  C symbols (build/libbeam.a absent; run make first)")

# ── Python public API ──────────────────────────────────────────────────────
# The recorded file is the signature of every public class member. Comparing
# it catches a moved default, a renamed keyword, or a method that quietly
# stopped being public -- none of which the test suite necessarily fails on.
def _api_snapshot():
    out = [f"__all__: {sorted(BEAM.__all__)}", f"__version__: {BEAM.__version__}"]
    for name in sorted(BEAM.__all__):
        o = getattr(BEAM, name)
        if inspect.isclass(o):
            out.append(f"\nclass {name}:")
            for m in sorted(dir(o)):
                if m.startswith("_") and not m.startswith("__"):
                    continue
                a = inspect.getattr_static(o, m, None)
                if isinstance(a, property):
                    out.append(f"  property {m}")
                elif callable(a):
                    try:
                        out.append(f"  def {m}{inspect.signature(a)}")
                    except (ValueError, TypeError):
                        out.append(f"  def {m}(?)")
        else:
            try:
                out.append(f"\ndef {name}{inspect.signature(o)}")
            except (ValueError, TypeError):
                out.append(f"\ndef {name}(?)")
    return [l for l in "\n".join(out).splitlines() if l.strip()]


recorded = [l for l in (B / "python_api.txt").read_text().splitlines() if l.strip()]
current = _api_snapshot()
if current == recorded:
    print(f"  OK    Python public API ({len(current)} signature lines)")
else:
    fail.append("python API")
    print("  FAIL  Python public API")
    only_new = [l for l in current if l not in recorded][:5]
    only_old = [l for l in recorded if l not in current][:5]
    for l in only_old: print(f"          gone : {l.strip()}")
    for l in only_new: print(f"          added: {l.strip()}")

# ── Header pointers ────────────────────────────────────────────────────────
# The headers here carry the reasoning, and the reasoning points at files:
# "see the definition in score.c for when this is safe" was the only statement
# of a precondition, and `score.c` did not exist. The comment is the medium
# this codebase chose for the "why", and a pointer that no longer resolves
# decays it silently -- a reader who goes to check finds nothing and cannot
# tell whether the claim or the pointer is stale.
#
# Every `*.c` named in a header must therefore resolve against the tree, with
# an allowlist for the ones that are deliberately historical: prose about what
# the code used to be. Adding to that list is a claim; deleting a file a header
# points at, without saying so, fails here.
hist = set()
for line in (B / "header_file_refs_historical.txt").read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#"):
        hist.add(line)
have = {p.name for p in (ROOT / "src").rglob("*.c")}
have |= {p.name for p in (ROOT / "tests").rglob("*.c")}
dangling, used_hist = [], set()
for hdr in sorted(list((ROOT / "include/beam").glob("*.h"))
                  + list((ROOT / "src/native/internal/beam").glob("*.h"))):
    for ln, line in enumerate(hdr.read_text().splitlines(), 1):
        for name in re.findall(r"\b[A-Za-z_0-9/]+\.c\b", line):
            if name.split("/")[-1] in have:
                continue
            key = f"{hdr.name}:{name}"
            if key in hist:
                used_hist.add(key)
                continue
            dangling.append(f"{hdr.name}:{ln}  {name}")
if dangling:
    fail.append("header pointers")
    print("  FAIL  header pointers: named a .c that does not exist")
    for d in dangling[:8]:
        print(f"          {d}")
    print("          Fix the pointer, or record it in "
          "baseline/header_file_refs_historical.txt if it is deliberate history.")
elif sorted(hist - used_hist):
    # A recorded exception that no longer appears is an exception nobody needs.
    fail.append("header pointers")
    print("  FAIL  header pointers: recorded history that is no longer mentioned")
    for k in sorted(hist - used_hist):
        print(f"          {k}")
else:
    print(f"  OK    header pointers ({len(hist)} historical mentions recorded)")

# ── Test inventory ─────────────────────────────────────────────────────────
# A subset of a passing suite also passes, so shrinkage is invisible unless
# the collected list is compared by name.
root = ROOT
# splitlines, not split(): a parametrised id can contain spaces, and
# whitespace-splitting silently truncates it to the first word.
_py_want_text = (B / "pytest_cases.txt").read_text(encoding="utf-8")
# The rootdir is pinned by python_collect_argv, because ids are numbered
# from it and the configuration that sets it now lives inside tests/.
out = subprocess.run(python_collect_argv(root), capture_output=True, text=True,
                     encoding="utf-8", cwd=root).stdout
py_want, py_got = python_inventory(_py_want_text, out)
# The same whole ids under the names the environment guard below and the
# record writer already use: both want the path, which is now compared too.
py_want_lines, py_got_lines = py_want, py_got


# ── Is this interpreter able to see the whole suite? ───────────────────────
# The inventory compares what THIS environment collects against a list
# recorded from another, so an environment that cannot import a test module
# reports its cases as deleted. `tests/suites/io/test_parquet_streaming.py`
# opens with `pytest.importorskip("pyarrow")` at MODULE level: without pyarrow
# the file yields nothing, `--collect-only` says "636 tests collected" instead
# of 642 with no error, no skip count and no mention of the file, and the
# check fails with six names under `missing`. The obvious response to that --
# re-record the baseline -- deletes them.
#
# Module level is what makes it invisible. Three other files call
# `importorskip` inside a test function; those cases are still collected and
# merely skip at run time, which the inventory sees correctly.
#
# So the environment is checked before the comparison, by a route that does
# not depend on pytest's output format: a file the baseline knows about, that
# is still on disk, and that produced nothing here, means this interpreter
# cannot collect it. That is a different fact from "the tests were removed"
# and it is reported differently.
silent = sorted({l.split("::", 1)[0] for l in py_want_lines}
                - {l.split("::", 1)[0] for l in py_got_lines}
                - {f for f in {l.split("::", 1)[0] for l in py_want_lines}
                   if not (root / f).exists()})
if silent:
    fail.append("pytest inventory (environment)")
    print("  FAIL  pytest inventory: this interpreter cannot collect the "
          "whole suite")
    for f in silent:
        print(f"          silent : {f}  (on disk, collected nothing here)")
    print(f"          running: {sys.executable}")
    print("          Re-recording the baseline from here would DELETE those "
          "cases. Run the gate under the environment that can import them; "
          "in this repo that is .venv/bin/python.")
else:
    check("pytest inventory", sorted(py_got), sorted(py_want))

c_want = (B / "c_cases.txt").read_text().split()   # C names have no spaces
# Asked of the binaries, not grepped out of their source: see
# check_lib.c_inventory. A binary that cannot be asked is reported by name
# and the comparison is NOT attempted, because a partial inventory measured
# against the whole baseline buries the one line that says why in 300 lines
# of "missing".
c_got, c_problems = c_inventory(root)
if c_problems:
    fail.append("C test inventory (build)")
    print("  FAIL  C test inventory: the binaries could not all be asked")
    for problem in c_problems:
        print(f"          {problem}")
    print("          Not compared against the baseline: an inventory short "
          "by a whole file would report every one of its tests as missing.")
else:
    check("C test inventory", c_got, c_want)

# ── The thread plan is actually in force ───────────────────────────────────
# Every analytics binding must resolve a thread plan for its call. The wiring
# is a macro around the GIL-release region, so "did all of them get it" is a
# question about SOURCE, and a static answer is stronger than a timing test:
# it cannot be passed by a lucky machine, and it fails on the next region
# somebody adds without the macro.
#
# Two things are checked, because each catches what the other cannot. A
# remaining `Py_BEGIN_ALLOW_THREADS` is a region that plans nothing. A `return`
# or `goto` inside a planned region is a path that skips the leave, which would
# apply one call's thread count to the next, unrelated call -- the macro's
# safety rests on there being no such path, so that is verified and not assumed.
PLANNED_BINDINGS = ("analytics_bindings.c", "common.c")
_bind = ROOT / "src" / "native" / "bindings" / "python"
_unplanned, _escapes, _regions = [], [], 0
for _name in PLANNED_BINDINGS:
    _lines = (_bind / _name).read_text(encoding="utf-8").splitlines()
    _depth = 0
    for _i, _ln in enumerate(_lines, 1):
        if "Py_BEGIN_ALLOW_THREADS" in _ln:
            _unplanned.append(f"{_name}:{_i}")
        if "BEAM_PY_PLANNED_BEGIN" in _ln:
            _depth, _regions = 1, _regions + 1
            continue
        if "BEAM_PY_PLANNED_END" in _ln:
            _depth = 0
            continue
        if _depth and ("return" in _ln.split("/*")[0].split("//")[0]
                       or "goto" in _ln.split("/*")[0].split("//")[0]):
            _escapes.append(f"{_name}:{_i}: {_ln.strip()}")
if _unplanned:
    fail.append("unplanned analytics region")
    print(f"  FAIL  {len(_unplanned)} analytics region(s) resolve no thread plan: "
          f"{', '.join(_unplanned)}")
elif _escapes:
    fail.append("escape from a planned region")
    print(f"  FAIL  {len(_escapes)} path(s) leave a planned region without "
          f"restoring the plan:")
    for _e in _escapes:
        print(f"          {_e}")
else:
    print(f"  OK    thread plan wiring ({_regions} analytics regions planned, "
          f"no escapes)")

# ── Numerical contract ─────────────────────────────────────────────────────
c = json.loads((B / "corpus.json").read_text())
seqs, ab = c["seqs"], c["ab"]
want = json.loads((B / "numeric.json").read_text())
got = {}
# Three orders, each under the default prior and under strengths five orders
# of magnitude apart between levels, so both extremes of borrowing are pinned.
_UNEVEN = {2: (0.01, 50.0, 3.0), 3: (0.01, 50.0, 3.0, 400.0),
           4: (0.01, 50.0, 3.0, 400.0, 0.2)}
for order in (2, 3, 4):
    for prior in ("default", "uneven"):
        alpha = None if prior == "default" else _UNEVEN[order]
        m = BEAM.BeamModel(seqs, abundances=ab, order=order, max_length=15,
                           alpha=alpha,
                           alpha_length=None if alpha is None else 0.5)
        t = f"o{order}/{prior}"
        got[t + "/scores"]   = [float(m.log_pseq(s)).hex() for s in seqs[:30]]
        got[t + "/unseen"]   = float(m.log_pseq("CWWWWWWWWF")).hex()
        got[t + "/D0"]       = float(m.path_count()).hex()
        got[t + "/hill"]     = [float(m.hill_number(q)).hex() for q in (0, 1, 2, 3)]
        got[t + "/entropy"]  = float(m.entropy()).hex()
        got[t + "/accmax"]   = float(m.accumulator_max()).hex()
        got[t + "/range"]    = [float(x).hex() for x in m.surprisal_range()]
        got[t + "/moments"]  = [float(x).hex() for x in np.ravel(m.surprisal_moments())]
        got[t + "/lenpmf"]   = [float(x).hex() for x in np.ravel(m.length_pmf())]
        got[t + "/topology"] = list(map(int, (m.topology(10)["n_nodes"],
                                              m.topology(10)["n_edges"])))
        got[t + "/sim"]      = sorted(m.simulate(15, seed=7).sequences)
        a = m.pseq_analysis()
        p, w = a.atoms(256).as_arrays()
        got[t + "/atoms"]    = [len(p), float(np.sum(w)).hex(),
                                float(p[0]).hex(), float(p[-1]).hex()]
        pm = BEAM.PublicnessModel([100, 500, 2000, 9000], depth_bins=4)
        got[t + "/public"]   = [float(x).hex() for x in
                                np.ravel(pm.expected_counts(p, w)["expected_counts"])]

# Exact hex comparison everywhere EXCEPT the publicness keys.
#
# The occupancy pipeline runs its forward transform through numpy's FFT, not
# through this library, and numpy's last bits depend on the numpy build: which
# pocketfft it ships, which SIMD path it dispatches to, how it threads. Two
# correct numpy versions disagree in the final ulp, which made this gate fail
# on a reviewer's machine while passing here. Demanding bit-identity of
# somebody else's FFT is not a property of BEAM, so those keys are compared
# with a relative tolerance and everything else stays exact.
FFT_TOL = 1e-9


def _close(a, b):
    fa, fb = float.fromhex(a), float.fromhex(b)
    if fa == fb:
        return True
    scale = max(abs(fa), abs(fb))
    return scale > 0 and abs(fa - fb) <= FFT_TOL * scale


bad = []
for k in sorted(got):
    ref = want.get(k)
    if got[k] == ref:
        continue
    if k.endswith("/public") and isinstance(ref, list) and len(ref) == len(got[k]) \
            and all(_close(x, y) for x, y in zip(got[k], ref)):
        continue
    bad.append(k)
if bad:
    fail.append("numeric")
    print(f"  FAIL  numeric ({len(bad)}/{len(got)} keys differ)")
    import os as _os
    if _os.environ.get("BEAM_NUMERIC_ALL"):
        same = [k for k in sorted(got) if k not in bad]
        print("        UNCHANGED (" + str(len(same)) + "): " + ", ".join(same))
    for k in bad[:8]:
        print(f"          {k}")
else:
    print(f"  OK    numeric ({len(got)} keys, order 2/3/4 x default/uneven prior; "
          f"publicness within {FFT_TOL:g} relative, being numpy's FFT)")

# ── Every public math method, exactly ──────────────────────────────────────
#
# `numeric` above pins four of the math quantities. These are the other
# twenty-two, and they exist because a refactor that moves math between
# modules can move a number without touching a test that asserts a property.
_math_got = math_capture()
_math_path = B / "math_values.json"
if not _math_path.exists():
    # The one run allowed to find no baseline is the one recording it. Any
    # other run has lost the comparison, which is a failure and not a note:
    # a clean gate over an absent net is worse than no net at all.
    if missing_baseline_is_fatal("math_values", sys.argv):
        fail.append("math values")
        print("  FAIL  math values: no baseline at "
              f"{_math_path.name} (record it with --only math_values)")
    else:
        print("  note  math values: no baseline yet, recording it now")
elif (_math_bad := math_differences(
        _math_got, json.loads(_math_path.read_text(encoding="utf-8")))):
    fail.append("math values")
    print(f"  FAIL  math values ({len(_math_bad)}/{len(_math_got)} keys differ)")
    for k in _math_bad[:8]:
        print(f"          {k}")
else:
    print(f"  OK    math values ({len(_math_got)} quantities over "
          f"{len({k.split('/')[0] for k in _math_got})} models, bit-identical)")

# ── Persistence: the saved bytes themselves ────────────────────────────────
m = BEAM.BeamModel(seqs, abundances=ab, order=4, max_length=15)
tmp = B / "_check_fixture.beam"
m.save(str(tmp.with_suffix("")))
_fixture_bytes = tmp.read_bytes()
fixture_size = len(_fixture_bytes)

# Two regions of a saved model are not a function of the corpus, so hashing
# them would recompare a clock against its own recording and fail every run.
# The provenance window is three build timestamps; the trailing CRC covers
# that window and moves with it. The format says where the first one is --
# nothing here hardcodes an offset -- and the check keeps its reach over
# every other byte, which is all of the payload and the rest of the header.
_prov_off, _prov_len = BEAM.file_provenance_span()
_hashed = (_fixture_bytes[:_prov_off]
           + _fixture_bytes[_prov_off + _prov_len:-4])
assert len(_hashed) == fixture_size - _prov_len - 4
digest = hashlib.sha256(_hashed).hexdigest()
del _fixture_bytes, _hashed
check("saved-model bytes", digest, want["fixture_sha256"])
tmp.unlink(missing_ok=True)

# ── Thread independence ────────────────────────────────────────────────────
tw = json.loads((B / "threads.json").read_text())
tg = {}
for n in (1, 4, 16):
    BEAM.set_threads(n)
    mm = BEAM.BeamModel(seqs, abundances=ab, order=4, max_length=15)
    tg[str(n)] = [[float(mm.log_pseq(s)).hex() for s in seqs[:40]],
                  float(mm.path_count()).hex(), float(mm.entropy()).hex(),
                  float(mm.hill_number(1)).hex()]
BEAM.set_threads(0)
check("thread independence", tg, tw)

# ── Correctness against enumeration ────────────────────────────────────────
# Everything above pins REPRODUCIBILITY: each quantity is compared against a
# recording of itself, so a quantity that has been wrong since it was written
# passes for ever. `exact.py` compares against brute-force enumeration of a
# model's whole support instead, which is the only check here that can fail on
# a defect that predates the baseline. Three such defects were found that way,
# all inside quantities the baseline already pinned.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import exact                                                     # noqa: E402

if exact.run():
    fail.append("exact")

# ── Recording ──────────────────────────────────────────────────────────────
# `--record TAG` writes every baseline from the values THIS run computed,
# rather than from a second copy of the computation. One derivation, two uses:
# a recorder that re-derives what it records can record something the gate
# never checked.
#
# Three refusals, because re-recording to make a failure go away is the hazard
# this whole directory exists to prevent:
#
#   - `exact` failing means the model disagrees with brute-force enumeration.
#     Every other check compares a quantity against a recording of itself, so
#     recording over them while the independent one fails would bake in a
#     defect and retire the only witness to it.
#   - the environment fault means this interpreter cannot collect the whole
#     suite, and recording would delete the cases it cannot see. The gate
#     already says so; the recorder must not proceed past it.
#   - a missing TAG, because the previous file is preserved as
#     `<name>.<TAG>.<ext>` and an unnamed one cannot be found again.
if "--record" in sys.argv:
    _i = sys.argv.index("--record")
    _tag = sys.argv[_i + 1] if len(sys.argv) > _i + 1 else ""
    if not _tag or _tag.startswith("-"):
        print("\nREFUSING TO RECORD: --record needs a TAG, which names the "
              "preserved copy of what is being replaced")
        sys.exit(2)
    if "exact" in fail:
        print("\nREFUSING TO RECORD: exact.py failed. It is the only check "
              "here with an independent reference, so recording over the "
              "others while it fails would pin a defect and retire its only "
              "witness.")
        sys.exit(2)
    if any(f.startswith("pytest inventory (environment)") for f in fail):
        print("\nREFUSING TO RECORD: this interpreter cannot collect the "
              "whole suite, so recording would delete the cases it cannot "
              "see.")
        sys.exit(2)

    # The fourth refusal, settled here with the others rather than between
    # two writes. It used to sit beside the c_cases write, by which point
    # python_api.txt and pytest_cases.txt had been overwritten and their
    # preserved copies consumed -- so a tree whose binaries were not built
    # lost two baselines before being told no, and could not re-run the same
    # tag afterwards. It is conditional because a short C inventory says
    # nothing about the Python one: `--only pytest_cases` is still recordable.
    _targets = record_targets(sys.argv)
    if "c_cases" in _targets:
        refuse_short_c_inventory(c_problems)

    def _write(name, text):
        path = B / name
        stem, _, ext = name.rpartition(".")
        prev = B / f"{stem}.{_tag}.{ext}"
        if path.exists():
            if prev.exists():
                print(f"  REFUSED {name}: {prev.name} already exists")
                sys.exit(2)
            prev.write_bytes(path.read_bytes())
        path.write_text(text)
        print(f"  recorded {name}  (previous kept as {prev.name})")

    print(f"\nRECORDING {', '.join(sorted(_targets))}")
    if "python_api" in _targets:
        _write("python_api.txt", "\n".join(current) + "\n")
    if "pytest_cases" in _targets:
        _write("pytest_cases.txt", "\n".join(sorted(py_got_lines)) + "\n")
    if "c_cases" in _targets:
        _write("c_cases.txt", "\n".join(sorted(c_got)) + "\n")
    if "numeric" in _targets:
        _num = dict(got)
        _num["fixture_sha256"] = digest
        _num["fixture_bytes"] = fixture_size
        _write("numeric.json", json.dumps(_num, indent=1, sort_keys=True) + "\n")
    if "threads" in _targets:
        _write("threads.json", json.dumps(tg, indent=1, sort_keys=True) + "\n")
    if "math_values" in _targets:
        _write("math_values.json",
               json.dumps(_math_got, indent=1, sort_keys=True) + "\n")
    print("\nRECORDED. Write the RERECORDED.md section naming every line "
          "that moved before committing this.")
    sys.exit(0)

print()
if fail:
    print(f"BASELINE CHECK FAILED: {', '.join(fail)}")
    sys.exit(1)
print("BASELINE CHECK PASSED")
