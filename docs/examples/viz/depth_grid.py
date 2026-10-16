"""Exact richness and two-sample overlap on a grid of depths, for the depth slider."""
import numpy as np
import BEAM
from BEAM.analytics.exact_occupancy import ExactOccupancy
from _docs import save_data

model = BEAM.load_pretrained("foundation")
occupancy = ExactOccupancy(model)

depths = np.logspace(3, 11, 161)            # 20 points per decade, up to 1e11
richness = [occupancy.richness(d) for d in depths]
sharing = [occupancy.sharing(d, d) for d in depths]
novelty = occupancy.novelty_many(depths)
deficit = occupancy.sampling_deficit_many(depths)
H = model.entropy()

save_data("depth_grid", {
    "depths": depths,
    "richness": [r.value for r in richness],
    "richness_bound": [r.bound for r in richness],
    "sharing": [s.value for s in sharing],
    "sharing_bound": [s.bound for s in sharing],
    "novelty": [u.value for u in novelty],
    "entropy": H,
    "sample_entropy": [H - x.value for x in deficit],
    # relative bounds: novelty directly; the sample entropy H - S from S's bound
    "novelty_bound": [u.bound for u in novelty],
    "sample_entropy_bound": [x.bound * x.value / (H - x.value) for x in deficit],
})
print(f"{len(depths)} depths; worst relative bound "
      f"{max(max(r.bound for r in richness), max(s.bound for s in sharing)):.0e}")
