import numpy as np
from BEAM import BeamModel
# cells_1, cells_2: two repertoires of one group
pooled = BeamModel(cells_1 + cells_2)                         # build from all the cells
merged = BeamModel(cells_1).merged(BeamModel(cells_2))        # or add up two built models

difference = np.abs(pooled.log_pseq_batch(probe) - merged.log_pseq_batch(probe)).max()
print(f"largest difference in log probability over 1,000 sequences: {difference:.1e}")
