from BEAM import BeamModel
from BEAM.analytics.distance import compare
# two repertoires of the same population (sequences s1, s2 and their clone sizes c1, c2):
# donor 1 has large expanded clones, donor 2 does not
weighted = compare(BeamModel(s1, abundances=c1, order=3),
                   BeamModel(s2, abundances=c2, order=3))
unique = compare(BeamModel(s1, order=3), BeamModel(s2, order=3))

print(f"distance, weighted by cells:          {weighted.distance:.4f}")
print(f"distance, each distinct sequence once: {unique.distance:.4f}")
