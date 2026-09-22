"""
Package-wide runtime, memory and validity profile for BEAM.

One invocation is one RUN. A run is stamped with the commit it measured, so a
regression can always be traced to a revision and reverted to it. Every case is
executed `--reps` times; the table records the mean and the spread, and the
RETURN VALUE of every case is written to disk so a later run can be checked for
numerical drift as well as for speed.

    .venv/bin/python tools/runtime_profile/profile.py --tiers tiny,small
    .venv/bin/python tools/runtime_profile/profile.py --tiers foundation
    .venv/bin/python tools/runtime_profile/compare.py <runA> <runB>

Layout produced:

    runs/<YYYYmmdd-HHMMSS>__<commit>/
        manifest.json      commit, date, machine, threads, dirty flag
        timings.csv        one row per (tier, case)
        results/           the captured return value of every case
    TIMINGS.md             wide table, one column per run
"""
import argparse, csv, hashlib, json, os, platform, subprocess, sys, threading
import time, traceback

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)

import numpy as np
import psutil
import BEAM
from BEAM import BeamModel, Builder
from cases import CASES

RUNS = os.path.join(HERE, "runs")
CACHE = os.path.join(HERE, "cache")
DONORS = os.path.join(ROOT, ".research/data_hub/10x_vdj")
AA = set("ACDEFGHIKLMNPQRSTVWY")


# ── data loading ────────────────────────────────────────────────
def load_donor(donor_dirs):
    """Real productive TRB CDR3s with UMI abundances from 10x annotations."""
    import csv as _csv
    from collections import Counter
    seqs = Counter()
    for d in donor_dirs:
        path = os.path.join(d, "all_contig_annotations.csv")
        with open(path) as fh:
            for r in _csv.DictReader(fh):
                if r.get("chain") != "TRB" or r.get("productive") != "True":
                    continue
                s = r.get("cdr3") or ""
                if 1 <= len(s) <= 27 and all(ch in AA for ch in s):
                    seqs[s] += int(r.get("umis") or 1)
    items = sorted(seqs.items())
    return [s for s, _ in items], [float(c) for _, c in items]


def tier_data(tier):
    ds = sorted(os.path.join(DONORS, d) for d in os.listdir(DONORS))
    if tier == "tiny":
        return load_donor(ds[:1])
    if tier == "small":
        return load_donor(ds)
    raise ValueError(tier)


# ── measurement ─────────────────────────────────────────────────
class PeakRSS:
    """Sample RSS while a call runs. Coarse, but it is the only way to see a
    transient allocation that is freed before the call returns -- which is
    exactly what the DP scratch buffers are."""

    def __init__(self, interval=0.005):
        self.interval, self.peak, self._stop = interval, 0, False

    def __enter__(self):
        self.proc = psutil.Process()
        self.base = self.proc.memory_info().rss
        self.peak = self.base
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()
        return self

    def _loop(self):
        while not self._stop:
            try:
                r = self.proc.memory_info().rss
                if r > self.peak:
                    self.peak = r
            except Exception:
                return
            time.sleep(self.interval)

    def __exit__(self, *a):
        self._stop = True
        self._t.join(timeout=1.0)


def timeit(fn, reps, warmup=1):
    """Run `fn` `reps` times after `warmup` untimed calls.

    The warmup is not cosmetic. The model's live-frontier cache is built on
    first use, so without it whichever case ran first paid for every case after
    it -- `top_k_20` measured 3.2 s and `top_k_100` 0.17 s purely because of
    the order they appear in. Timings here are WARM.
    """
    times, val, peak = [], None, 0
    for _ in range(warmup):
        val = fn()
    for _ in range(reps):
        with PeakRSS() as p:
            t0 = time.perf_counter()
            val = fn()
            dt = time.perf_counter() - t0
        times.append(dt)
        peak = max(peak, p.peak - p.base)
    return times, val, peak


def timing(t, reps):
    """The five numbers a timing row carries.

    MINIMUM is the honest one and the comparison keys on it. A run can only be
    slowed by the machine -- a scheduler preemption, a page fault, another
    process -- never sped up by it, so the fastest observed run is the closest
    thing to the work itself and the mean is that plus a noise tail.

    This matters here rather than being a generality. `top_k` varies 1.82x
    between repetitions of one build on one corpus while `entropy` varies
    1.14x, and with three reps and a mean the profile reported `top_k` 3.3x
    slower across a model change that a controlled A/B showed had made it
    FASTER. `max_s` is kept so a comparison can say how much of a difference
    its own noise could account for.
    """
    t = [float(x) for x in t]
    return dict(mean_s=float(np.mean(t)), std_s=float(np.std(t)),
                min_s=min(t), max_s=max(t), reps=reps)


def fixed(value):
    """A row that is a measurement rather than a timing: no spread to report."""
    return dict(mean_s=value, std_s=0.0, min_s=value, max_s=value, reps=1)


# ── result capture ──────────────────────────────────────────────
def capture(value, kind, outdir, tag):
    """Normalise a return value into something comparable across runs.

    Arrays go to .npy so a later comparison is numerical rather than textual;
    everything else lands in JSON. `opaque` records only the type, for objects
    whose value has no stable representation.
    """
    os.makedirs(outdir, exist_ok=True)

    def norm(v, path):
        if isinstance(v, np.ndarray):
            f = os.path.join(outdir, f"{tag}{path}.npy")
            np.save(f, v)
            fin = v[np.isfinite(v)] if v.dtype.kind == "f" else v
            return {"__array__": os.path.basename(f), "shape": list(v.shape),
                    "dtype": str(v.dtype),
                    "sum": float(fin.sum()) if fin.size else 0.0,
                    "min": float(fin.min()) if fin.size else None,
                    "max": float(fin.max()) if fin.size else None,
                    "n_nonfinite": int(v.size - fin.size)
                    if v.dtype.kind == "f" else 0}
        if isinstance(v, (np.floating, float)):
            return float(v)
        if isinstance(v, (np.integer, int)):
            return int(v)
        if isinstance(v, (bool, np.bool_)):
            return bool(v)
        if v is None or isinstance(v, str):
            return v
        if isinstance(v, dict):
            return {str(k): norm(v[k], f"{path}.{k}") for k in sorted(v, key=str)}
        if isinstance(v, (list, tuple)):
            if v and all(isinstance(x, (int, float, np.number)) for x in v):
                return norm(np.asarray(v), path)
            return [norm(x, f"{path}[{i}]") for i, x in enumerate(v)]
        return {"__repr__": type(v).__name__}

    if kind == "opaque":
        return {"__opaque__": type(value).__name__}
    return norm(value, "")


# ── run bookkeeping ─────────────────────────────────────────────
def git_info():
    def g(*a):
        try:
            return subprocess.check_output(["git", *a], cwd=ROOT,
                                           text=True).strip()
        except Exception:
            return "?"
    dirty = g("status", "--porcelain", "--untracked-files=no")
    return dict(commit=g("rev-parse", "HEAD"),
                short=g("rev-parse", "--short", "HEAD"),
                subject=g("log", "-1", "--pretty=%s"),
                date=g("log", "-1", "--pretty=%cI"),
                dirty=bool(dirty))


def build_context(tier, m, seqs, ab):
    """Everything the cases need besides the model."""
    class Ctx:
        pass
    c = Ctx()
    c.tier, c.m = tier, m
    rng = np.random.default_rng(20260827)
    probe = seqs if len(seqs) <= 200_000 else [
        seqs[i] for i in rng.choice(len(seqs), 200_000, replace=False)]
    c.probe = probe
    c.probe_big = probe[:100_000]
    lens = list(m.observed_lengths)
    c.mid_length = int(lens[len(lens) // 2]) if lens else 14
    lo, hi = m.surprisal_range()
    c.grid = np.linspace(lo, min(hi, lo + 200.0), 64)
    c.holdout_seqs = seqs[: max(1, len(seqs) // 100)]

    # Real per-repertoire sampling depths, for the occupancy model. Cached on
    # the context because building one is itself a timed case.
    c._pub = None

    def publicness():
        if c._pub is None:
            from BEAM import PublicnessModel
            c._pub = PublicnessModel(_depths())
        return c._pub
    c.publicness = publicness
    return c


_DEPTHS = None


def _depths():
    """`unique_sequences` per repertoire, from the release manifest."""
    global _DEPTHS
    if _DEPTHS is not None:
        return _DEPTHS
    import csv as _csv
    p = os.path.join(ROOT, ".research/data_hub/release_20260812_censored/"
                           "tcrb_human_repertoires_20260812.tsv")
    vals = []
    with open(p) as fh:
        for r in _csv.DictReader(fh, delimiter="\t"):
            try:
                v = float(r["unique_sequences"])
            except (TypeError, ValueError, KeyError):
                continue
            if v > 0:
                vals.append(v)
    _DEPTHS = np.asarray(sorted(vals)[:2000], dtype=np.float64)
    return _DEPTHS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiers", default="tiny,small")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--repeat", type=int, default=1,
                    help="run the whole case list this many times and keep "
                         "the best of all timings per case. Separate batches "
                         "do not share allocator, cache or frequency state, "
                         "which consecutive reps do -- so this is what makes "
                         "a volatile case callable. Costs N times the clock.")
    ap.add_argument("--order", type=int, default=4)
    ap.add_argument("--threads", type=int, default=0, help="0 = library default")
    ap.add_argument("--only", default="", help="comma-separated case names")
    ap.add_argument("--skip", default="", help="comma-separated case names")
    ap.add_argument("--label", default="", help="extra tag for the run id")
    ap.add_argument("--case-timeout", type=float, default=0,
                    help="seconds; 0 = no limit (advisory, recorded only)")
    a = ap.parse_args()

    if a.threads:
        BEAM.set_threads(a.threads)

    gi = git_info()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    rid = f"{stamp}__{gi['short']}" + (f"__{a.label}" if a.label else "")
    outdir = os.path.join(RUNS, rid)
    resdir = os.path.join(outdir, "results")
    os.makedirs(resdir, exist_ok=True)

    manifest = dict(
        run_id=rid, started=time.strftime("%Y-%m-%dT%H:%M:%S"),
        git=gi, reps=a.reps, repeat=a.repeat, order=a.order,
        tiers=a.tiers.split(","),
        threads_configured=a.threads or None,
        threads_effective=BEAM.effective_threads(),
        machine=dict(node=platform.node(), system=platform.system(),
                     release=platform.release(), python=platform.python_version(),
                     cpu_count=os.cpu_count(),
                     ram_total_bytes=psutil.virtual_memory().total),
    )
    with open(os.path.join(outdir, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)
    print(f"run {rid}  commit {gi['short']} ({gi['subject'][:50]})"
          f"{'  [DIRTY]' if gi['dirty'] else ''}")

    only = {x for x in a.only.split(",") if x}
    skip = {x for x in a.skip.split(",") if x}
    rows = []

    for tier in manifest["tiers"]:
        print(f"\n=== tier {tier} ===", flush=True)
        # ---- construction, timed as first-class cases ----
        if tier == "foundation":
            cached = os.path.join(CACHE, "foundation.beam")
            if not os.path.exists(cached):
                print(f"  no cached foundation at {cached}; "
                      f"run build_foundation.py first. Skipping tier.")
                continue
            info = json.load(open(cached + ".json"))
            for nm, key in (("build_observe", "t_observe_s"),
                            ("build_finalize", "t_finalize_s"),
                            ("save", "t_save_s")):
                rows.append(dict(tier=tier, group="lifecycle", case=nm,
                                 **fixed(info[key]),
                                 peak_rss_bytes=info.get("peak_rss_bytes", 0),
                                 status="cached"))
            t, m, pk = timeit(lambda: BeamModel.from_file(cached), a.reps)
            rows.append(dict(tier=tier, group="lifecycle", case="load",
                             **timing(t, a.reps), peak_rss_bytes=pk, status="ok"))
            rows.append(dict(tier=tier, group="lifecycle", case="file_bytes",
                             **fixed(0.0),
                             peak_rss_bytes=info["file_bytes"], status="ok",
                             value=info["file_bytes"]))
            seqs, ab = [], []
            # a probe set drawn from the corpus the model was built on
            import pyarrow.parquet as pq
            pf = pq.ParquetFile(os.path.join(
                ROOT, ".research/data_hub/release_20260812_censored/"
                "tcrb_human_cdr3_abundance_productive_maxlen27_censored.parquet"))
            b0 = next(pf.iter_batches(batch_size=200_000, columns=["cdr3_aa"]))
            seqs = b0.column("cdr3_aa").to_pylist()
            ob = Builder(order=a.order)
            ob.observe(seqs[:20_000], None)
            other = ob.build()
        else:
            seqs, ab = tier_data(tier)
            print(f"  {len(seqs):,} sequences", flush=True)

            def do_build():
                b = Builder(order=a.order)
                b.observe(seqs, ab)
                return b.build()
            # split observe / finalize so the dense trellis sweep is visible
            b = Builder(order=a.order)
            t, _, pk = timeit(lambda: b.observe(seqs, ab), 1)
            rows.append(dict(tier=tier, group="lifecycle", case="build_observe",
                             **fixed(t[0]),
                             peak_rss_bytes=pk, status="ok"))
            ts = []
            for _ in range(a.reps):
                bb = Builder(order=a.order)
                bb.observe(seqs, ab)
                t0 = time.perf_counter()
                m = bb.build()
                ts.append(time.perf_counter() - t0)
            rows.append(dict(tier=tier, group="lifecycle", case="build_finalize",
                             **timing(ts, a.reps), peak_rss_bytes=0, status="ok"))
            t, _, pk = timeit(lambda: do_build(), a.reps)
            rows.append(dict(tier=tier, group="lifecycle", case="build_total",
                             **timing(t, a.reps), peak_rss_bytes=pk, status="ok"))

            path = os.path.join(CACHE, f"{tier}.beam")
            os.makedirs(CACHE, exist_ok=True)
            t, _, pk = timeit(lambda: m.save(path), a.reps)
            rows.append(dict(tier=tier, group="lifecycle", case="save",
                             **timing(t, a.reps), peak_rss_bytes=pk, status="ok"))
            fb = os.path.getsize(path)
            rows.append(dict(tier=tier, group="lifecycle", case="file_bytes",
                             **fixed(0.0), peak_rss_bytes=fb,
                             status="ok", value=fb))
            t, _, pk = timeit(lambda: BeamModel.from_file(path), a.reps)
            rows.append(dict(tier=tier, group="lifecycle", case="load",
                             **timing(t, a.reps), peak_rss_bytes=pk, status="ok"))
            # A 10% cohort, so `subtract` models leave-one-cohort-out rather
            # than emptying the model -- which the library rightly refuses.
            k = max(1, len(seqs) // 10)
            ob = Builder(order=a.order)
            ob.observe(seqs[:k], ab[:k])
            other = ob.build()

        ctx = build_context(tier, m, seqs, ab)
        ctx.other = other

        # ── batches ───────────────────────────────────────────────
        #
        # `--repeat N` runs the WHOLE case list N times and keeps the best of
        # all N*reps timings per case, rather than taking N*reps consecutive
        # calls once.
        #
        # The distinction is the whole point. Repetitions inside one visit to
        # a case share that visit's state -- allocator arenas, page cache, CPU
        # frequency, whatever the cases before it left behind -- so they agree
        # with each other while the batch as a whole is off. Measured on two
        # runs of identical code: `spectrum_counting` was tight to 1.03x
        # across its three consecutive reps in one run and 4.79x slower than
        # the other run, and no within-run statistic can see that. Separate
        # batches can, because they do not share the state.
        #
        # It costs N times the wall clock, so it is opt-in, and one batch
        # (the default) behaves exactly as before.
        acc = {}
        for batch in range(max(1, a.repeat)):
            if a.repeat > 1:
                print(f"  -- batch {batch + 1}/{a.repeat}", flush=True)
            for spec in CASES:
                nm = spec["name"]
                if tier not in spec["tiers"]:
                    continue
                if only and nm not in only:
                    continue
                if nm in skip:
                    continue
                reps = spec.get("reps_by_tier", {}).get(tier, a.reps)
                try:
                    tb, val, pkb = timeit(lambda: spec["fn"](ctx), reps)
                except Exception as e:
                    acc[nm] = acc.get(nm) or dict(t=[], val=None, pk=0, err=e,
                                                  spec=spec)
                    acc[nm]["err"] = e
                    continue
                e = acc.setdefault(nm, dict(t=[], val=None, pk=0, err=None,
                                            spec=spec))
                e["t"].extend(tb)
                e["val"] = val
                e["pk"] = max(e["pk"], pkb)

        for nm, e in acc.items():
            spec = e["spec"]
            reps = len(e["t"]) or spec.get("reps_by_tier", {}).get(tier, a.reps)
            try:
                if e["err"] is not None and not e["t"]:
                    raise e["err"]
                t, val, pk = e["t"], e["val"], e["pk"]
                cap = capture(val, spec["kind"], resdir, f"{tier}__{nm}")
                with open(os.path.join(resdir, f"{tier}__{nm}.json"), "w") as fh:
                    json.dump(cap, fh, indent=1, sort_keys=True, default=str)
                rows.append(dict(tier=tier, group=spec["group"], case=nm,
                                 **timing(t, reps),
                                 peak_rss_bytes=pk, status="ok"))
                # The spread is printed, not buried, because a case whose
                # own repetitions disagree cannot support a claim about a
                # change between runs -- and `top_k` is such a case.
                spread = max(t) / min(t) if min(t) > 0 else 1.0
                flag = f"  ~{spread:.2f}x spread" if spread >= 1.25 else ""
                print(f"  {nm:26s} {min(t)*1e3:10.2f} ms  "
                      f"peak +{pk/2**20:8.1f} MB{flag}", flush=True)
            except Exception as e:
                rows.append(dict(tier=tier, group=spec["group"], case=nm,
                                 mean_s=float("nan"), std_s=float("nan"),
                                 min_s=float("nan"), max_s=float("nan"), reps=0, peak_rss_bytes=0,
                                 status=f"ERROR: {type(e).__name__}: {e}"))
                with open(os.path.join(resdir, f"{tier}__{nm}.error.txt"),
                          "w") as fh:
                    fh.write(traceback.format_exc())
                print(f"  {nm:26s}   ERROR {type(e).__name__}: {e}", flush=True)

    fields = ["tier", "group", "case", "mean_s", "std_s", "min_s",
              "max_s", "reps",
              "peak_rss_bytes", "status", "value"]
    with open(os.path.join(outdir, "timings.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    manifest["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    manifest["n_cases"] = len(rows)
    manifest["n_errors"] = sum(1 for r in rows
                               if str(r["status"]).startswith("ERROR"))
    with open(os.path.join(outdir, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)

    print(f"\nwrote {outdir}")
    print(f"{len(rows)} cases, {manifest['n_errors']} errors")
    import render_table
    render_table.main()


if __name__ == "__main__":
    main()
