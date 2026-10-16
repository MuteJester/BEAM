"""Weighted vs unique models of one repertoire as clonal expansion grows."""
import numpy as np
import BEAM
from BEAM import BeamModel
from BEAM.analytics.distance import compare
from _docs import save_data
from _standins import population, expand

ref = BEAM.load_pretrained("foundation")
cells = population(ref, "A", 30_000, seed=7)
unique = BeamModel(sorted(set(cells)), order=3)
rows = []
for strength in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0):
    seqs, counts = expand(cells, seed=8, strength=strength)
    weighted = BeamModel(seqs, abundances=counts, order=3)
    top = seqs[int(np.argmax(counts))]
    order = np.argsort(counts)[::-1][:10]
    rows.append({
        "strength": strength,
        "top_clone": top,
        "top_share": max(counts) / sum(counts),
        "top10_share": float(np.sum(np.array(counts)[order]) / sum(counts)),
        "p_top_weighted": float(weighted.pseq(top)),
        "p_top_unique": float(unique.pseq(top)),
        "distance": compare(weighted, unique).distance,
        "length_weighted": {int(L): p for L, p in weighted.length_distribution.items() if 8 <= L <= 22},
        "length_unique": {int(L): p for L, p in unique.length_distribution.items() if 8 <= L <= 22},
    })
save_data("expansion", {"rows": rows})
print(f"{len(rows)} expansion strengths")
