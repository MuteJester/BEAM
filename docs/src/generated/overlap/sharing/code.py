import BEAM
from BEAM.analytics.exact_occupancy import ExactOccupancy

model = BEAM.load_pretrained("human_tcrb_cdr3_aa_foundation")
occupancy = ExactOccupancy(model)

print("  depth A    depth B   shared sequences   share of A's distinct sequences")
for a, b in [(1e5, 1e5), (1e5, 1e6), (1e6, 1e6), (1e7, 1e7)]:
    shared = occupancy.sharing(a, b).value
    distinct_a = occupancy.richness(a).value
    print(f"{a:9,.0f}  {b:9,.0f}  {shared:17,.0f}  {shared / distinct_a:32.1%}")

