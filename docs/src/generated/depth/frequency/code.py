import BEAM
from BEAM.analytics.exact_occupancy import ExactOccupancy

model = BEAM.load_pretrained("human_tcrb_cdr3_aa_foundation")
occupancy = ExactOccupancy(model)

depth = 1_000_000
print(f"in {depth:,} cells, expected number of sequences seen exactly j times:")
for j in [1, 2, 3, 4, 5, 10]:
    print(f"  j = {j:2}   {occupancy.frequency(depth, j).value:12,.0f}")
