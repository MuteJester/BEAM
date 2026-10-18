import numpy as np
from BEAM import BeamModel
from BEAM.analytics.distance import compare
# --- hide
import BEAM
from _standins import population
ref = BEAM.load_pretrained("human_tcrb_cdr3_aa_foundation")
train = {g: [s for seed in seeds for s in population(ref, g, 30_000, seed=seed)]
         for g, seeds in {"A": (71, 72, 73), "B": (74, 75, 76), "C": (77, 78, 79)}.items()}
new_donor = {"A": population(ref, "A", 30_000, seed=81),
             "B": population(ref, "B", 30_000, seed=82),
             "C": population(ref, "C", 30_000, seed=83)}
# --- show
# train[g]: all cells of group g's repertoires; new_donor[g]: a repertoire not used to build
groups = "ABC"
models = {g: BeamModel(train[g], order=3) for g in groups}    # one model per group

print("one sequence at a time  (share of a new donor's sequences most probable under each group)")
for true in groups:
    seqs = sorted(set(new_donor[true]))
    scores = np.vstack([models[g].log_pseq_batch(seqs) for g in groups])
    called = scores.argmax(axis=0)
    shares = "  ".join(f"{g}: {np.mean(called == i):.0%}" for i, g in enumerate(groups))
    print(f"  donor from {true}:  {shares}")

print("whole repertoire  (distance from the donor's model to each group's model)")
for true in groups:
    donor = BeamModel(new_donor[true], order=3)
    d = {g: compare(donor, models[g]).distance for g in groups}
    nearest = min(d, key=d.get)
    print(f"  donor from {true}:  " + "  ".join(f"{g}: {d[g]:.4f}" for g in groups) + f"   nearest: {nearest}")
