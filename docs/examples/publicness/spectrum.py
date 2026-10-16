import numpy as np
import BEAM
from BEAM import PublicnessModel
from _docs import figure  # hide

model = BEAM.load_pretrained("foundation")
p, w = model.pseq_analysis().atoms(512).as_arrays()   # every sequence, grouped by probability
cohort = PublicnessModel([100_000] * 100)              # 100 people, 100,000 cells each

counts = cohort.expected_counts(p, w)["expected_counts"]   # [k]: sequences in exactly k people
seen = counts[1:].sum()
print(f"distinct sequences in the cohort: {seen:,.0f}")
print(f"private (in one person):          {counts[1]:,.0f}  ({counts[1] / seen:.1%})")
for k in (2, 5, 10, 50, 100):
    print(f"in exactly {k:3} people:            {counts[k]:,.1f}")
# --- hide
with figure("spectrum", size=(6.4, 3.4)) as (fig, ax):
    k = np.arange(1, len(counts))
    ax.loglog(k, counts[1:], "o", ms=3)
    ax.set_xlabel("number of people carrying a sequence (out of 100)")
    ax.set_ylabel("expected number of sequences")
# --- show
