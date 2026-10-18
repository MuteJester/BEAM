import numpy as np
import BEAM
from BEAM.analytics.exact_occupancy import ExactOccupancy

model = BEAM.load_pretrained("human_tcrb_cdr3_aa_foundation")
occupancy = ExactOccupancy(model)

depths = np.logspace(4, 6, 10)            # ten donors, 10,000 to 1,000,000 cells
result = occupancy.private_at_depth(depths, total_depth=depths.sum())

print("   cells   distinct   private to this donor")
for d, n, priv in zip(depths, result["distinct"], result["private"]):
    print(f"{d:9,.0f}  {n.value:9,.0f}   {priv.value / n.value:8.1%}")
