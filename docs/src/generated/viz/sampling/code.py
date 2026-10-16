"""Seeded draws and the exact length distribution, for the sampling animation."""
import BEAM
from _docs import save_data

model = BEAM.load_pretrained("foundation")
draws = model.simulate(3000, seed=17)
lengths = model.length_distribution
save_data("sampling", {
    "sequences": draws.sequences,
    "log_p": [round(float(v), 3) for v in draws.log_probs],
    "length_distribution": {int(L): p for L, p in lengths.items() if p > 1e-6},
})
print(f"{len(draws.sequences)} draws")
