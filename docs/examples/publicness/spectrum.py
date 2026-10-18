import numpy as np
import BEAM
from BEAM import PublicnessModel
from BEAM.analytics.exact_occupancy import ExactOccupancy
from _docs import figure  # hide

BEAM.set_memory_budget(32 << 30)       # a fine probability grid needs about 20 GB
model = BEAM.load_pretrained("human_tcrb_cdr3_aa_foundation")

# Every sequence the model can produce, grouped onto a fine grid of probabilities
p, w = model.pseq_analysis().atoms(16384).as_arrays()
cohort = PublicnessModel([100_000] * 100)             # 100 people, 100,000 cells each

counts = cohort.expected_counts(p, w)["expected_counts"]   # [k]: sequences in exactly k people
seen = counts[1:].sum()
print(f"distinct sequences in the cohort: {seen:,.0f}")
print(f"private (in one person):          {counts[1]:,.0f}  ({counts[1] / seen:.1%})")
for k in (2, 5, 10, 50, 100):
    print(f"in exactly {k:3} people:            {counts[k]:,.1f}")

# The grid is an approximation; check it against the grid-free calculation
direct = ExactOccupancy(model).equal_depth_cohort(100_000, 100)
print(f"grid-free: {direct['seen'].value:,.0f} distinct, {direct['private'].value:,.0f} private "
      f"(grid within {abs(seen / direct['seen'].value - 1):.1e} and "
      f"{abs(counts[1] / direct['private'].value - 1):.1e})")
# --- hide
# As in the paper's spectrum panel: sequences per unit publicness on log-spaced
# bins, the top bin (sequences in nearly everyone, collected at k = 100) left out.
n_people = len(counts) - 1
edges = np.unique(np.round(np.logspace(0, np.log10(n_people + 1), 17)).astype(int))
centres, per_unit = [], []
for lo, hi in zip(edges[:-2], edges[1:-1]):
    ks = np.arange(lo, hi)
    centres.append(np.exp(np.log(ks).mean()))
    per_unit.append(counts[lo:hi].sum() / len(ks))
with figure("spectrum", size=(6.4, 3.4)) as (fig, ax):
    ax.loglog(centres, per_unit, "o-", ms=4)
    ax.set_xlabel("number of people carrying a sequence (out of 100)")
    ax.set_ylabel("expected sequences\nper unit publicness")
# --- show
