import BEAM
from BEAM.analytics.exact_occupancy import ExactOccupancy

model = BEAM.load_pretrained("foundation")
occupancy = ExactOccupancy(model)

print("   cells sampled   distinct sequences   distinct share   error bound")
for depth in [1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9]:
    r = occupancy.richness(depth)
    print(f"{depth:16,.0f}  {r.value:19,.0f}  {r.value / depth:15.1%}  {r.bound:12.0e}")

