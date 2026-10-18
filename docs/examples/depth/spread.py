import BEAM
from BEAM.analytics.exact_occupancy import ExactOccupancy

model = BEAM.load_pretrained("human_tcrb_cdr3_aa_foundation")
occupancy = ExactOccupancy(model)

print("   cells sampled   expected distinct   standard deviation   as share of expected")
for depth in [10_000, 100_000, 1_000_000, 10_000_000]:
    mean = occupancy.richness(depth)
    sd = occupancy.richness_sd(depth)      # computed, like the mean: no simulation
    print(f"{depth:16,}  {mean.value:18,.0f}  {sd.value:19,.1f}  {sd.value / mean.value:21.3%}")
