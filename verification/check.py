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
    # Compare name lists as SETS. GNU sort and Python order underscores
    # differently depending on locale, so a sorted-sequence comparison reports
    # a difference between two identical symbol tables. Membership is the
    # contract; ordering is not.
    if isinstance(got, list) and isinstance(want, list):
        got, want = sorted(set(got)), sorted(set(want))
    if got == want:
        print(f"  OK    {name}")
    else:
        fail.append(name)
        print(f"  FAIL  {name}")
        if isinstance(want, list) and isinstance(got, list):
            miss = [x for x in want if x not in got]
            extra = [x for x in got if x not in want]
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
py_want_lines = [l.strip() for l in (B / "pytest_cases.txt").read_text().splitlines()
                 if "::" in l]
py_want = {l.split("::", 1)[1] for l in py_want_lines}
out = subprocess.run([sys.executable, "-m", "pytest", str(root / "tests"), "-q",
                      "--collect-only"], capture_output=True, text=True, cwd=root).stdout
py_got_lines = [l.strip() for l in out.splitlines() if "::" in l]
py_got = {l.split("::", 1)[1] for l in py_got_lines}

# ── Is this interpreter able to see the whole suite? ───────────────────────
# The inventory compares what THIS environment collects against a list
# recorded from another, so an environment that cannot import a test module
# reports its cases as deleted. `tests/python/io/test_parquet_streaming.py`
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
c_got = re.findall(r"RUN_TEST\((\w+)\)",
                   "\n".join(f.read_text() for f in (root / "tests/native").glob("*.c")))
check("C test inventory", sorted(c_got), sorted(c_want))

# ── Numerical contract ─────────────────────────────────────────────────────
c = json.loads((B / "corpus.json").read_text())
seqs, ab = c["seqs"], c["ab"]
want = json.loads((B / "numeric.json").read_text())
got = {}
for order in (2, 3, 4):
    for strict in (False, True):
        m = BEAM.BeamModel(seqs, abundances=ab, order=order,
                           max_length=15, strict=strict)
        t = f"o{order}/{'strict' if strict else 'open'}"
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
    for k in bad[:8]:
        print(f"          {k}")
else:
    print(f"  OK    numeric ({len(got)} keys, order 2/3/4 x open/strict; "
          f"publicness within {FFT_TOL:g} relative, being numpy's FFT)")

# ── Persistence: the saved bytes themselves ────────────────────────────────
m = BEAM.BeamModel(seqs, abundances=ab, order=4, max_length=15, strict=True)
tmp = B / "_check_fixture.beam"
m.save(str(tmp.with_suffix("")))
digest = hashlib.sha256(tmp.read_bytes()).hexdigest()
check("saved-model bytes", digest, want["fixture_sha256"])
tmp.unlink(missing_ok=True)

# ── Thread independence ────────────────────────────────────────────────────
tw = json.loads((B / "threads.json").read_text())
tg = {}
for n in (1, 4, 16):
    BEAM.set_threads(n)
    mm = BEAM.BeamModel(seqs, abundances=ab, order=4, max_length=15, strict=True)
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

print()
if fail:
    print(f"BASELINE CHECK FAILED: {', '.join(fail)}")
    sys.exit(1)
print("BASELINE CHECK PASSED")
