"""
Compare two runs: speed, memory, and -- the part that matters -- whether the
answers moved.

    .venv/bin/python tools/runtime_profile/compare.py <runA> <runB>
    .venv/bin/python tools/runtime_profile/compare.py --last 2

A case is a REGRESSION if it got slower, and a BREAKAGE if its value changed
beyond tolerance. The second is the one that should stop a merge.
"""
import argparse, csv, glob, json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")


def load(rid):
    d = rid if os.path.isdir(rid) else os.path.join(RUNS, rid)
    rows = {(r["tier"], r["case"]): r
            for r in csv.DictReader(open(os.path.join(d, "timings.csv")))}
    man = json.load(open(os.path.join(d, "manifest.json")))
    return d, man, rows


def read_capture(d, tier, case):
    p = os.path.join(d, "results", f"{tier}__{case}.json")
    if not os.path.exists(p):
        return None
    return json.load(open(p))


def walk(obj, prefix=""):
    """Flatten a captured value into {path: scalar-or-array-marker}."""
    out = {}
    if isinstance(obj, dict):
        if "__array__" in obj:
            out[prefix] = obj
            return out
        if "__opaque__" in obj or "__repr__" in obj:
            return out
        for k in sorted(obj):
            out.update(walk(obj[k], f"{prefix}.{k}"))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.update(walk(v, f"{prefix}[{i}]"))
    elif isinstance(obj, (int, float, bool)) or obj is None or isinstance(obj, str):
        out[prefix] = obj
    return out


def cmp_values(dA, dB, tier, case, rtol, atol):
    """Return (verdict, detail). verdict in {same, drift, CHANGED, missing}."""
    a, b = read_capture(dA, tier, case), read_capture(dB, tier, case)
    if a is None or b is None:
        return "missing", ""
    fa, fb = walk(a), walk(b)
    if set(fa) != set(fb):
        only = (set(fa) ^ set(fb))
        return "CHANGED", f"structure differs at {sorted(only)[:3]}"

    worst, worst_path = 0.0, ""
    for k in fa:
        va, vb = fa[k], fb[k]
        if isinstance(va, dict) and "__array__" in va:
            try:
                A = np.load(os.path.join(dA, "results", va["__array__"]))
                B = np.load(os.path.join(dB, "results", vb["__array__"]))
            except Exception as e:
                return "CHANGED", f"array unreadable at {k}: {e}"
            if A.shape != B.shape:
                return "CHANGED", f"shape {A.shape} vs {B.shape} at {k}"
            if A.dtype.kind == "f":
                mA = np.isfinite(A); mB = np.isfinite(B)
                if not np.array_equal(mA, mB):
                    return "CHANGED", f"non-finite pattern differs at {k}"
                if mA.any():
                    d = np.abs(A[mA] - B[mB])
                    scale = np.maximum(np.abs(A[mA]), np.abs(B[mB]))
                    rel = np.where(scale > 0, d / np.maximum(scale, 1e-300), d)
                    r = float(rel.max())
                    if r > worst:
                        worst, worst_path = r, k
            elif not np.array_equal(A, B):
                return "CHANGED", f"integer array differs at {k}"
        elif isinstance(va, float) or isinstance(vb, float):
            if va is None or vb is None:
                if va != vb:
                    return "CHANGED", f"{k}: {va} vs {vb}"
                continue
            fa_, fb_ = float(va), float(vb)
            if np.isfinite(fa_) != np.isfinite(fb_):
                return "CHANGED", f"{k}: {va} vs {vb}"
            if np.isfinite(fa_):
                sc = max(abs(fa_), abs(fb_), 1e-300)
                r = abs(fa_ - fb_) / sc
                if r > worst:
                    worst, worst_path = r, k
        else:
            if va != vb:
                return "CHANGED", f"{k}: {va!r} vs {vb!r}"

    if worst > rtol:
        return "CHANGED", f"max rel diff {worst:.3e} at {worst_path}"
    if worst > 0:
        return "drift", f"max rel diff {worst:.3e} at {worst_path}"
    return "same", ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--last", type=int, default=0)
    ap.add_argument("--rtol", type=float, default=1e-9)
    ap.add_argument("--atol", type=float, default=0.0)
    ap.add_argument("--slower-than", type=float, default=1.05,
                    help="flag a case slower than this ratio")
    ap.add_argument("--floor-ms", type=float, default=10.0,
                    help="cases faster than this are noise-dominated and are "
                         "reported separately rather than ranked")
    ap.add_argument("--sigma", type=float, default=2.0,
                    help="a difference must exceed this many combined standard "
                         "deviations to count as real")
    a = ap.parse_args()

    if a.last:
        ds = sorted(glob.glob(os.path.join(RUNS, "*")))[-a.last:]
        if len(ds) < 2:
            print("need two runs"); return 1
        rids = [os.path.basename(d) for d in ds[-2:]]
    elif len(a.runs) == 2:
        rids = a.runs
    else:
        print("give two run ids, or --last 2"); return 1

    dA, mA, rA = load(rids[0])
    dB, mB, rB = load(rids[1])
    print(f"A = {rids[0]}   commit {mA['git']['short']}  {mA['git']['subject'][:44]}")
    print(f"B = {rids[1]}   commit {mB['git']['short']}  {mB['git']['subject'][:44]}")
    print()

    keys = sorted(set(rA) | set(rB))
    # A run of one batch cannot report a volatile case honestly.
    #
    # Timings inside one visit to a case share that visit's state, so they
    # agree with each other while the batch as a whole is off: measured on two
    # runs of identical code, `spectrum_counting` was tight to 1.03x across
    # its reps and 4.79x apart between the runs. `--repeat N` in profile.py is
    # what separates the batches; without it the spread column understates the
    # uncertainty and a difference here may be the batch rather than the code.
    reps_ok = True
    for man, rid in ((mA, rids[0]), (mB, rids[1])):
        if int(man.get("repeat") or 1) < 2:
            reps_ok = False
    if not reps_ok:
        print("  NOTE  at least one run used a single batch (--repeat 1), so "
              "a volatile case's\n        difference here may be the batch "
              "and not the code. Re-run both with\n        --repeat 3 before "
              "trusting any single case.\n")

    faster, slower, changed, errs, unresolved = [], [], [], [], []
    print(f"{'tier':<11} {'case':<26} {'A':>11} {'B':>11} {'speed':>9}  value")
    print("-" * 88)
    for k in keys:
        ra, rb = rA.get(k), rB.get(k)
        if not ra or not rb:
            continue
        sa, sb = str(ra["status"]), str(rb["status"])
        if sa.startswith("ERROR") or sb.startswith("ERROR"):
            errs.append((k, sa, sb))
            continue
        # MINIMUM, not mean. The machine can only slow a repetition down, so
        # the fastest one is the closest thing to the work itself; the mean is
        # that plus however long the scheduler happened to steal. Older runs
        # have no `min_s`, so they fall back to what they recorded.
        try:
            ta = float(ra.get("min_s") or ra["mean_s"])
            tb = float(rb.get("min_s") or rb["mean_s"])
        except (TypeError, ValueError):
            continue
        if not (ta > 0 and tb > 0):
            continue
        ratio = ta / tb
        verdict, detail = cmp_values(dA, dB, k[0], k[1], a.rtol, a.atol)
        if verdict == "CHANGED":
            changed.append((k, detail))

        # A timing difference is only reported when it is BOTH large enough to
        # matter and larger than the noise. Sub-10ms cases here are dominated
        # by OpenMP region entry, which swings 30x between runs of identical
        # code -- ranking those alongside real results is how a benchmark
        # invents findings.
        try:
            sa, sb = float(ra["std_s"] or 0), float(rb["std_s"] or 0)
        except (TypeError, ValueError):
            sa = sb = 0.0
        noise = a.sigma * ((sa * sa + sb * sb) ** 0.5)
        # A single repetition has no variance estimate, so `noise` is 0 and
        # every difference looks significant. Those cases cannot be called.
        try:
            single = int(ra.get("reps") or 0) < 2 or int(rb.get("reps") or 0) < 2
        except (TypeError, ValueError):
            single = True
        # And a case whose OWN repetitions disagree by more than the change
        # being claimed cannot support the claim, however many sigma a
        # three-sample standard deviation says it is. `top_k` spans 1.82x
        # between repetitions of one build on one corpus, and that is what
        # let this tool report it 3.3x slower across a change that a
        # controlled A/B showed had made it faster.
        def spread(r):
            try:
                lo, hi = float(r.get("min_s") or 0), float(r.get("max_s") or 0)
                return hi / lo if lo > 0 else 1.0
            except (TypeError, ValueError):
                return 1.0
        own = max(spread(ra), spread(rb))

        if single:
            if ratio >= a.slower_than or ratio <= 1 / a.slower_than:
                unresolved.append((k, ratio, "single rep"))
        elif max(ratio, 1 / ratio) <= own:
            if ratio >= a.slower_than or ratio <= 1 / a.slower_than:
                unresolved.append((k, ratio, f"within its own {own:.2f}x spread"))
        elif max(ta, tb) * 1e3 < a.floor_ms:
            if ratio >= a.slower_than or ratio <= 1 / a.slower_than:
                unresolved.append((k, ratio, "below floor"))
        elif abs(ta - tb) <= noise:
            if ratio >= a.slower_than or ratio <= 1 / a.slower_than:
                unresolved.append((k, ratio, "within noise"))
        elif ratio >= a.slower_than:
            faster.append((k, ratio))
        elif ratio <= 1 / a.slower_than:
            slower.append((k, 1 / ratio))
        mark = {"same": "ok", "drift": "~", "CHANGED": "**CHANGED**",
                "missing": "?"}[verdict]
        print(f"{k[0]:<11} {k[1]:<26} {ta*1e3:>9.2f}ms {tb*1e3:>9.2f}ms "
              f"{ratio:>8.2f}x  {mark} {detail[:28]}")

    print("\n" + "=" * 88)
    print(f"faster (>= {a.slower_than}x): {len(faster)}")
    for k, r in sorted(faster, key=lambda x: -x[1])[:12]:
        print(f"   {r:6.2f}x  {k[0]}/{k[1]}")
    print(f"SLOWER: {len(slower)}")
    for k, r in sorted(slower, key=lambda x: -x[1])[:12]:
        print(f"   {r:6.2f}x slower  {k[0]}/{k[1]}")
    print(f"VALUE CHANGED: {len(changed)}")
    for k, d in changed[:20]:
        print(f"   {k[0]}/{k[1]}: {d}")
    print(f"unresolved (too fast or too noisy to call): {len(unresolved)}")
    for k, r, why in sorted(unresolved, key=lambda x: -abs(1 - x[1]))[:8]:
        print(f"   {r:6.2f}x  {k[0]}/{k[1]}  ({why})")
    print(f"errors: {len(errs)}")
    for k, sa, sb in errs[:10]:
        print(f"   {k[0]}/{k[1]}  A={sa[:34]}  B={sb[:34]}")
    return 1 if changed else 0


if __name__ == "__main__":
    sys.exit(main())
