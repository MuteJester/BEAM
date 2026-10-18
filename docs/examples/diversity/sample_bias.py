import math
import BEAM
from BEAM.analytics.exact_occupancy import ExactOccupancy
from _docs import figure  # hide

model = BEAM.load_pretrained("human_tcrb_cdr3_aa_foundation")
occupancy = ExactOccupancy(model)
H = model.entropy()

depths = [1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10]
deficit = occupancy.sampling_deficit_many(depths)     # entropy lost to finite sampling
novelty = occupancy.novelty_many(depths)              # chance the next cell is a new sequence

print("   cells sampled   sample entropy   share of H   chance next cell is new")
for n, s, u in zip(depths, deficit, novelty):
    print(f"{n:16,.0f}   {H - s.value:10.3f} nats   {(H - s.value) / H:10.1%}   {u.value:22.1%}")
# --- hide
import numpy as np
grid = np.logspace(4, 11, 29)
d = occupancy.sampling_deficit_many(grid)
u = occupancy.novelty_many(grid)
with figure("bias", size=(6.4, 3.4)) as (fig, ax):
    ax.semilogx(grid, [(H - x.value) / H for x in d], lw=2, label="sample entropy, share of the model's")
    ax.semilogx(grid, [x.value for x in u], lw=2, label="chance the next cell is new")
    ax.set_xlabel("cells sampled")
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False)
# --- show
