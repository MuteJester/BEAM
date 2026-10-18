"""Simulated counts of distinct sequences in 100,000 cells, against the computed value."""
import numpy as np
import BEAM
from BEAM.analytics.exact_occupancy import ExactOccupancy
from _docs import figure, save_data

model = BEAM.load_pretrained("human_tcrb_cdr3_aa_foundation")
DEPTH = 100_000
occupancy = ExactOccupancy(model)
computed = occupancy.richness(DEPTH)
sd = occupancy.richness_sd(DEPTH)          # the spread of one sample around it, computed too

# 100 independent samples of 100,000 cells, each counted.
simulated = np.array([len(set(model.sample(DEPTH, seed=s))) for s in range(1, 101)])

save_data("simulated", {"depth": DEPTH, "computed": computed.value, "bound": computed.bound,
                        "sd": sd.value, "sd_bound": sd.bound, "simulated": simulated})
print(f"computed:  {computed.value:,.1f} ± {sd.value:.1f} (one standard deviation; "
      f"relative error bounds {computed.bound:.1g} and {sd.bound:.1g})")
print(f"simulated: mean {simulated.mean():,.1f}, standard deviation {simulated.std(ddof=1):.1f}, "
      f"range {simulated.min():,} to {simulated.max():,}")
inside = np.abs(simulated - computed.value) <= 2 * sd.value
print(f"simulations within two computed standard deviations: {inside.sum()} of {simulated.size}")

runs = np.arange(1, simulated.size + 1)
with figure("estimates", size=(6.4, 3.2)) as (fig, ax):
    ax.axhspan(computed.value - 2 * sd.value, computed.value + 2 * sd.value, color="#438f91",
               alpha=0.15, lw=0, label="computed ± 2 standard deviations")
    ax.plot(runs, simulated, "o", ms=3.5, label="one simulation each")
    ax.plot(runs, np.cumsum(simulated) / runs, lw=1.5, label="mean of the simulations so far")
    ax.axhline(computed.value, color="#438f91", lw=2, label="computed by BEAM")
    ax.set_xlabel("simulation")
    ax.set_ylabel("distinct sequences\nin 100,000 cells")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
