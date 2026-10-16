import numpy as np
import BEAM
from BEAM import BeamModel

reference = BEAM.load_pretrained("foundation")          # 38,538 repertoires: the population
# --- hide
from _standins import population
donors = [population(reference, "B", 30_000, seed=s) for s in (1, 2, 3)]
# --- show
# donors: three repertoires from one group (stand-in population B: longer CDR3s)
per_repertoire = [BeamModel(cells) for cells in donors]   # one model per repertoire
group = BeamModel([s for cells in donors for s in cells])  # the group's cells, pooled


def long_share(model):
    """P(length >= 17) under a model."""
    return sum(p for L, p in model.length_distribution.items() if L >= 17)


for name, model in [("donor 1", per_repertoire[0]), ("donor 2", per_repertoire[1]),
                    ("donor 3", per_repertoire[2]), ("group", group), ("reference", reference)]:
    print(f"{name:10} P(length >= 17) = {long_share(model):.3f}   sequences seen: {model.n_observed:>11,}")
