import numpy as np
import BEAM
from BEAM import PublicnessModel
from BEAM.analytics.exact_occupancy import ExactOccupancy

BEAM.set_memory_budget(32 << 30)       # a fine probability grid needs about 20 GB
model = BEAM.load_pretrained("foundation")

# Every sequence the model can produce, grouped onto a fine grid of probabilities
p, w = model.pseq_analysis().atoms(16384).as_arrays()
cohort = PublicnessModel([100_000] * 100)             # 100 people, 100,000 cells each

counts = cohort.expected_counts(p, w)["expected_counts"]   # [k]: sequences in exactly k people
seen = counts[1:].sum()
print(f"distinct sequences in the cohort: {seen:,.0f}")
print(f"private (in one person):          {counts[1]:,.0f}  ({counts[1] / seen:.1%})")
for k in (2, 5, 10, 50, 100):
    print(f"in exactly {k:3} people:            {counts[k]:,.1f}")

# The grid is an approximation; check it against the exact, grid-free calculation
exact = ExactOccupancy(model).equal_depth_cohort(100_000, 100)
print(f"exact: {exact['seen'].value:,.0f} distinct, {exact['private'].value:,.0f} private "
      f"(grid within {abs(seen / exact['seen'].value - 1):.1e} and "
      f"{abs(counts[1] / exact['private'].value - 1):.1e})")
