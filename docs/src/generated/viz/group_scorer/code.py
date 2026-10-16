"""Log probabilities of example sequences under three stand-in group models."""
import numpy as np
import BEAM
from BEAM import BeamModel
from _docs import save_data
from _standins import population, has_motif

ref = BEAM.load_pretrained("foundation")
seeds = {"A": (71, 72, 73), "B": (74, 75, 76), "C": (77, 78, 79)}
models = {g: BeamModel([s for seed in ss for s in population(ref, g, 30_000, seed=seed)])
          for g, ss in seeds.items()}
rng = np.random.default_rng(5)
picked = []
for g, seed in zip("ABC", (81, 82, 83)):
    pool = sorted(set(population(ref, g, 5_000, seed=seed)))
    picked += [(g, s) for s in rng.choice(pool, size=5, replace=False)]
rows = []
for g, s in picked:
    logp = {k: float(models[k].pseq(s, log=True)) for k in "ABC"}
    rows.append({"sequence": s, "drawn_from": g, "length": len(s), "motif": has_motif(s), "log_p": logp})
save_data("group_scorer", {"motif": "PG", "rows": rows})
print(f"{len(rows)} sequences")
