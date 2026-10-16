from BEAM import BeamModel, Mixture
# --- hide
import BEAM
reference = BEAM.load_pretrained("foundation")
donor_a = BeamModel(reference.sample(5_000, seed=72), order=2, max_length=20)
donor_b = BeamModel(reference.sample(5_000, seed=73), order=2, max_length=20)
# --- show

group = Mixture([donor_a, donor_b])            # pick a donor, then a cell
print(f"Hill number q=2: {group.hill_number(2.0):.6g}  (exact)")

try:
    group.hill_number(1.0)                     # BEAM has no exact method at q = 1
except Exception as err:
    print(f"q=1 refused: {type(err).__name__}")

bounds = group.entropy_bounds()                # what can be said exactly instead
print(bounds)
