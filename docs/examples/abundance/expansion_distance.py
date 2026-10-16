from BEAM import BeamModel
from BEAM.analytics.distance import compare
# --- hide
import BEAM
from _standins import population, expand
ref = BEAM.load_pretrained("foundation")
s1, c1 = expand(population(ref, "A", 30_000, seed=61), seed=62, strength=2.5)   # expanded
s2, c2 = expand(population(ref, "A", 30_000, seed=63), seed=64, strength=0)     # not expanded
# --- show
# two repertoires of the same population: donor 1 has large expanded clones, donor 2 does not
weighted = compare(BeamModel(s1, abundances=c1, order=3), BeamModel(s2, abundances=c2, order=3))
unique = compare(BeamModel(s1, order=3), BeamModel(s2, order=3))

print(f"distance, weighted by cells:          {weighted.distance:.4f}")
print(f"distance, each distinct sequence once: {unique.distance:.4f}")
