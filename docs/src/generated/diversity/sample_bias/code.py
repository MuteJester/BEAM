import math
import BEAM
from BEAM.analytics.exact_occupancy import ExactOccupancy

model = BEAM.load_pretrained("foundation")
occupancy = ExactOccupancy(model)
H = model.entropy()

depths = [1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10]
deficit = occupancy.sampling_deficit_many(depths)     # entropy lost to finite sampling
novelty = occupancy.novelty_many(depths)              # chance the next cell is a new sequence

print("   cells sampled   sample entropy   share of H   chance next cell is new")
for n, s, u in zip(depths, deficit, novelty):
    print(f"{n:16,.0f}   {H - s.value:10.3f} nats   {(H - s.value) / H:10.1%}   {u.value:22.1%}")
