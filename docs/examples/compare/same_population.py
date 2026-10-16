import BEAM
from BEAM import BeamModel
from BEAM.analytics.distance import compare

# --- hide
reference = BEAM.load_pretrained("foundation")
donor_a = BeamModel(reference.sample(50_000, seed=51))
donor_a2 = BeamModel(reference.sample(50_000, seed=52))
# --- show
# donor_a, donor_a2: two samples of 50,000 cells from the same population
result = compare(donor_a, donor_a2)
print(f"distance: {result.distance:.4f}")
print(f"from lengths: {result.length_squared_distance / result.squared_distance:.1%}"
      f"   from local patterns: {result.pattern_squared_distance / result.squared_distance:.1%}")
