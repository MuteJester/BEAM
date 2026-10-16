"""Log probabilities of held-out example sequences under three stand-in group models."""
import numpy as np
import BEAM
from BEAM import BeamModel
from _docs import save_data
from _standins import population, has_motif

ref = BEAM.load_pretrained("foundation")
seeds = {"A": (71, 72, 73), "B": (74, 75, 76), "C": (77, 78, 79)}
train = {g: [s for seed in ss for s in population(ref, g, 30_000, seed=seed)] for g, ss in seeds.items()}
models = {g: BeamModel(cells, order=3) for g, cells in train.items()}   # as in groups/which_group
seen = set().union(*train.values())

rng = np.random.default_rng(5)
rows, accuracy = [], {}
for g, seed in zip("ABC", (81, 82, 83)):
    unseen = sorted(set(population(ref, g, 30_000, seed=seed)) - seen)   # never seen by any model
    scores = np.vstack([models[k].log_pseq_batch(unseen) for k in "ABC"])
    accuracy[g] = float(np.mean(scores.argmax(axis=0) == "ABC".index(g)))
    # a stratified choice: two that carry the group's built-in difference where it has one
    marked = [s for s in unseen if (g == "B" and len(s) >= 17) or (g == "C" and has_motif(s))]
    rest = [s for s in unseen if s not in set(marked)]
    pick = list(rng.choice(marked, size=2, replace=False)) if marked else []
    pick += list(rng.choice(rest, size=5 - len(pick), replace=False))
    for s in pick:
        rows.append({"sequence": s, "drawn_from": g, "length": len(s), "motif": has_motif(s),
                     "log_p": {k: float(models[k].pseq(s, log=True)) for k in "ABC"}})
save_data("group_scorer", {"motif": "PG", "rows": rows, "accuracy": accuracy})
print(f"{len(rows)} sequences; per-sequence accuracy " + ", ".join(f"{g}: {a:.0%}" for g, a in accuracy.items()))
