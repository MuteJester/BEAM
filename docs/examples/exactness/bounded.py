from BEAM import BeamModel
from BEAM.analytics.exact_occupancy import ExactOccupancy
# --- hide
import BEAM
model = BeamModel(BEAM.load_pretrained("human_tcrb_cdr3_aa_foundation").sample(20_000, seed=71), order=2)
# --- show

occupancy = ExactOccupancy(model)
r = occupancy.richness(50_000)
print(r)                        # the value, its guaranteed bound, and how it was computed
print(f"{r.value:,.1f} distinct sequences, relative error bound {r.bound:.2g}")

try:
    float(r)                    # refused: the bound must not be dropped by accident
except TypeError as err:
    print("float():", err)
