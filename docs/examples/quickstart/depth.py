import BEAM
from BEAM.analytics.exact_occupancy import ExactOccupancy

model = BEAM.load_pretrained("human_tcrb_cdr3_aa_foundation")
occupancy = ExactOccupancy(model)

distinct = occupancy.richness(100_000)
print(f"distinct sequences in 100,000 TCRs: {distinct.value:,.0f}")
print(f"guaranteed relative error bound:   {distinct.bound:.2g}")
