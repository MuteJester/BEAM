import numpy as np  # hide
import BEAM
from BEAM.analytics.exact_occupancy import ExactOccupancy
from _docs import figure  # hide

model = BEAM.load_pretrained("foundation")
occupancy = ExactOccupancy(model)

print("   cells sampled   distinct sequences   distinct share   error bound")
for depth in [1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9]:
    r = occupancy.richness(depth)
    print(f"{depth:16,.0f}  {r.value:19,.0f}  {r.value / depth:15.1%}  {r.bound:12.0e}")

# --- hide
depths = np.logspace(3, 10, 36)
values = [occupancy.richness(d).value for d in depths]
with figure("saturation", size=(6.4, 3.4)) as (fig, ax):
    ax.loglog(depths, depths, "--", color="#7d7d7d", lw=1, label="every cell a new sequence")
    ax.loglog(depths, values, lw=2, label="expected distinct sequences")
    ax.set_xlabel("cells sampled (depth)")
    ax.set_ylabel("distinct sequences")
    ax.legend(frameon=False)
# --- show
