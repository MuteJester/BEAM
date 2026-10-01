"""
Measure the DENSE code path, which the standard tiers never touch.

`state_weights_dense` runs only when `sparse` is false, which takes an
explicit `sparse=False`. The default Python configuration and the foundation
model are both sparse, so the whole runtime profile exercises
`state_weights_sparse` and says nothing at all about the dense one.

The dense x-hoist is supposed to pay 2.60x single-threaded on a full layer.
If it does, it shows up here and nowhere else.

The sequences are a JSON list of [sequence, count] pairs, named by the
environment variable BEAM_BENCH_SEQS:

    BEAM_BENCH_SEQS=tcrb.json python bench_dense.py [n] [reps] [threads]
"""
import json, os, statistics, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

import BEAM
from BEAM import BeamModel

SEQS = os.environ.get("BEAM_BENCH_SEQS", "")


def timeit(fn, reps, warmup=1):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return statistics.mean(ts), (statistics.pstdev(ts) if len(ts) > 1 else 0.0)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    threads = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    if threads:
        BEAM.set_threads(threads)

    if not SEQS:
        sys.exit("set BEAM_BENCH_SEQS to a JSON list of [sequence, count] pairs")
    data = json.load(open(SEQS))[:n]
    seqs = [s for s, _ in data]
    ab = [float(c) for _, c in data]

    m = BeamModel(seqs, abundances=ab, order=4, sparse=False)
    assert not m.config["sparse"], "expected the dense path"
    lens = list(m.observed_lengths)
    mid = int(lens[len(lens) // 2])

    cases = {
        "build_finalize": lambda: BeamModel(seqs, abundances=ab, order=4,
                                            strict=False),
        "log_partition_all": lambda: [m.log_partition(L) for L in lens],
        "entropy": lambda: m.entropy(),
        "path_count": lambda: m.path_count(mid),
        "topology": lambda: m.topology(mid),
        "surprisal_range": lambda: m.surprisal_range(),
        "spectrum": lambda: m.spectrum(bins=256),
        "sample_2k": lambda: m.sample(2000, seed=7),
    }
    print(f"# dense (strict=False) model, {len(seqs):,} seqs, "
          f"threads={BEAM.effective_threads()}, reps={reps}")
    out = {}
    for name, fn in cases.items():
        try:
            mu, sd = timeit(fn, reps)
            out[name] = (mu, sd)
            print(f"{name:<20}{mu*1e3:>10.2f} ms  +/- {sd*1e3:.2f}", flush=True)
        except Exception as e:
            print(f"{name:<20}   ERROR {type(e).__name__}: {e}", flush=True)
    json.dump({k: v for k, v in out.items()},
              open(os.path.join(HERE, f"dense_{os.environ.get('TAG','x')}.json"),
                   "w"), indent=1)


if __name__ == "__main__":
    main()
