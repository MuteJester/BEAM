"""Exact carrier distributions for the publicness explorer."""
import numpy as np
import BEAM
from BEAM import PublicnessModel
from _docs import save_data

model = BEAM.load_pretrained("human_tcrb_cdr3_aa_foundation")
probabilities = np.logspace(-10, -3, 71)        # 10 points per decade
examples = ["CASSLGQGYEQYF", "CASSLGQGAYNEQFF", "CASSPRDRGLAGGYNEQFF", "CASSIRSSYEQYF"]
cohorts = {f"{n}x{d}": [d] * n for n in (10, 100, 500) for d in (10_000, 100_000, 1_000_000)}


def trimmed(pmf):
    """The pmf from its first to last entry above 1e-7, with that offset."""
    keep = np.nonzero(pmf > 1e-7)[0]
    lo, hi = (int(keep[0]), int(keep[-1])) if keep.size else (0, 0)
    return {"k0": lo, "p": [float(f"{v:.5g}") for v in pmf[lo:hi + 1]]}


out = {"probabilities": probabilities,
       "examples": {s: float(model.pseq(s)) for s in examples},
       "cohorts": {}}
for key, depths in cohorts.items():
    cohort = PublicnessModel(depths)
    grid = [trimmed(cohort.pmf(p)) for p in probabilities]
    ex = {s: trimmed(cohort.pmf(p)) for s, p in out["examples"].items()}
    means = cohort.moments(np.array(list(probabilities) + list(out["examples"].values())))[0]
    out["cohorts"][key] = {"n": len(depths), "depth": depths[0], "grid": grid,
                           "examples": ex, "means": means}
save_data("publicness_grid", out)
print(f"{len(cohorts)} cohorts x {len(probabilities)} probabilities + {len(examples)} sequences")
