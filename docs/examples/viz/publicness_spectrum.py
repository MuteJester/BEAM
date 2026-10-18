"""Expected publicness spectra for several cohort sizes and depths."""
import numpy as np
import BEAM
from BEAM import PublicnessModel
from _docs import save_data

BEAM.set_memory_budget(32 << 30)
model = BEAM.load_pretrained("human_tcrb_cdr3_aa_foundation")
p, w = model.pseq_analysis().atoms(16384).as_arrays()   # fine grid: see publicness/spectrum
out = {}
for n in (10, 100, 1000):
    for depth in (10_000, 100_000, 1_000_000):
        counts = PublicnessModel([depth] * n).expected_counts(p, w)["expected_counts"]
        out[f"{n}x{depth}"] = {"n": n, "depth": depth,
                               "counts": [float(f"{c:.5g}") for c in counts[1:]]}
save_data("publicness_spectrum", out)
print(f"{len(out)} cohorts")
