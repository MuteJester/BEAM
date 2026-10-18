import numpy as np  # hide
import BEAM
from BEAM.analytics.exact_occupancy import ExactOccupancy
from _docs import figure  # hide

model = BEAM.load_pretrained("human_tcrb_cdr3_aa_foundation")
occupancy = ExactOccupancy(model)

print("  depth A    depth B   shared sequences   share of A's distinct sequences")
for a, b in [(1e5, 1e5), (1e5, 1e6), (1e6, 1e6), (1e7, 1e7)]:
    shared = occupancy.sharing(a, b).value
    distinct_a = occupancy.richness(a).value
    print(f"{a:9,.0f}  {b:9,.0f}  {shared:17,.0f}  {shared / distinct_a:32.1%}")

# --- hide
depths = np.logspace(3, 9, 31)
shared = [occupancy.sharing(d, d).value for d in depths]
distinct = [occupancy.richness(d).value for d in depths]
with figure("overlap", size=(6.4, 3.4)) as (fig, ax):
    ax.semilogx(depths, [s / r for s, r in zip(shared, distinct)], lw=2)
    ax.set_xlabel("cells sampled in each of two samples")
    ax.set_ylabel("share of one sample's sequences\nalso found in the other")
    ax.yaxis.set_major_formatter(__import__("matplotlib").ticker.PercentFormatter(1.0))
# --- show
