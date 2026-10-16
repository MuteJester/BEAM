"""Exact richness and two-sample overlap on a grid of depths, for the depth slider."""
import numpy as np
import BEAM
from BEAM.analytics.exact_occupancy import ExactOccupancy
from _docs import save_data

model = BEAM.load_pretrained("foundation")
occupancy = ExactOccupancy(model)

depths = np.logspace(3, 10, 141)            # 20 points per decade
richness = [occupancy.richness(d) for d in depths]
sharing = [occupancy.sharing(d, d) for d in depths]

save_data("depth_grid", {
    "depths": depths,
    "richness": [r.value for r in richness],
    "richness_bound": [r.bound for r in richness],
    "sharing": [s.value for s in sharing],
    "sharing_bound": [s.bound for s in sharing],
})
print(f"{len(depths)} depths; worst relative bound "
      f"{max(max(r.bound for r in richness), max(s.bound for s in sharing)):.0e}")
