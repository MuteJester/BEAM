import BEAM
from BEAM.analytics.exact_occupancy import ExactOccupancy

model = BEAM.load_pretrained("human_tcrb_cdr3_aa_foundation")
occupancy = ExactOccupancy(model)

print("   from depth    to depth   new distinct sequences   share of the extra cells")
for depth in [1e5, 1e6, 1e7, 1e8]:
    gained = occupancy.richness_difference(2 * depth, depth)
    print(f"{depth:13,.0f}  {2 * depth:10,.0f}  {gained.value:23,.0f}  {gained.value / depth:24.1%}")
