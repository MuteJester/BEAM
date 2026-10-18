import numpy as np
from BEAM import BeamModel
# --- hide
import BEAM
from _standins import population
reference = BEAM.load_pretrained("human_tcrb_cdr3_aa_foundation")
cells_1 = population(reference, "A", 20_000, seed=21)
cells_2 = population(reference, "A", 20_000, seed=22)
probe = reference.sample(1_000, seed=23)
# --- show
# cells_1, cells_2: two repertoires of one group
pooled = BeamModel(cells_1 + cells_2)                         # build from all the cells
merged = BeamModel(cells_1).merged(BeamModel(cells_2))        # or add up two built models

difference = np.abs(pooled.log_pseq_batch(probe) - merged.log_pseq_batch(probe)).max()
print(f"largest difference in log probability over 1,000 sequences: {difference:.1e}")
