"""Report how BEAM behaves on this machine, for comparison across platforms.

Checks that OpenMP is live (more than one thread where the machine has more
than one core) and that one thread and every thread give bit-identical
results, as BEAM promises. Then writes a fixed set of model quantities, each
at full precision, to a JSON file: run on two platforms, the two files show
how far their answers differ.

    python tools/release/platform_report.py report.json
    python tools/release/platform_report.py --compare linux.json windows.json macos.json

With --max-rel, comparing fails when any value of a later report differs from
the first by more than that relative amount.

Uses only BEAM and NumPy, and a model built from a fixed synthetic corpus.
Comparing needs neither: only the standard library.
"""
import argparse
import json
import os
import sys

ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
DEPTHS = [1e3, 1e5, 1e7, 1e9]


def corpus(seed, n):
    """CDR3-like strings: CAS... plus a random middle, F at the end."""
    import numpy as np
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        middle = "".join(rng.choice(list(ALPHABET), size=int(rng.integers(5, 12))))
        out.append("CAS" + middle + "F")
    return out


def quantities():
    from BEAM import BeamModel
    from BEAM.analytics.distance import compare
    from BEAM.analytics.exact_occupancy import ExactOccupancy
    a = BeamModel(corpus(1, 3000), order=3)
    b = BeamModel(corpus(2, 3000), order=3)
    probes = corpus(3, 40)
    occ = ExactOccupancy(a)
    return {
        "log_pseq": [float(v) for v in a.log_pseq_batch(probes)],
        "length_distribution": {str(k): float(v) for k, v in a.length_distribution.items()},
        "entropy": float(a.entropy()),
        "hill_2": float(a.hill_number(2.0)),
        "richness": [float(occ.richness(d).value) for d in DEPTHS],
        "sharing": [float(occ.sharing(d, d).value) for d in DEPTHS],
        "distance": float(compare(a, b).distance),
    }


def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        elif isinstance(v, list):
            out.update({f"{key}[{i}]": x for i, x in enumerate(v)})
        else:
            out[key] = v
    return out


def report(path):
    import BEAM
    cores = os.cpu_count() or 1
    threads = BEAM.effective_threads()
    print(f"BEAM {BEAM.__version__}, {sys.platform}, {cores} cores, "
          f"{threads} OpenMP threads")
    if cores > 1 and threads < 2 and "OMP_NUM_THREADS" not in os.environ:
        sys.exit("OpenMP is not active: one thread on a machine with several cores")

    BEAM.set_threads(1)
    single = quantities()
    BEAM.set_threads(0)
    parallel = quantities()
    if single != parallel:
        differ = [k for k, v in flatten(single).items() if flatten(parallel)[k] != v]
        sys.exit(f"one thread and {threads} threads disagree: {differ[:5]}")
    print(f"one thread and {threads} threads agree bit for bit")

    with open(path, "w") as f:
        json.dump({"platform": sys.platform, "values": parallel}, f, indent=1)
    print(f"wrote {path}")


def compare_reports(first, second):
    """The largest relative difference between two reports' values."""
    a, b = (json.load(open(p)) for p in (first, second))
    fa, fb = flatten(a["values"]), flatten(b["values"])
    print(f"{a['platform']} against {b['platform']}: {len(fa)} values")
    worst, identical = 0.0, 0
    for k in sorted(fa):
        x, y = fa[k], fb[k]
        if x == y:
            identical += 1
            continue
        rel = abs(x - y) / max(abs(x), abs(y), 1e-300)
        worst = max(worst, rel)
        print(f"  {k:28} {x!r:>24} {y!r:>24}  rel {rel:.1e}")
    print(f"{identical} of {len(fa)} identical; largest relative difference {worst:.1e}")
    return worst


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--compare", action="store_true",
                    help="compare reports with the first instead of writing one")
    ap.add_argument("--max-rel", type=float,
                    help="with --compare, fail above this relative difference")
    args = ap.parse_args()
    if args.compare:
        if len(args.paths) < 2:
            sys.exit("--compare takes at least two reports")
        worst = max(compare_reports(args.paths[0], p) for p in args.paths[1:])
        if args.max_rel is not None and worst > args.max_rel:
            sys.exit(f"largest relative difference {worst:.1e} exceeds {args.max_rel:.1e}")
    else:
        report(args.paths[0])


if __name__ == "__main__":
    main()
