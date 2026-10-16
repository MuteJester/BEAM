import BEAM
from BEAM import BeamModel
from BEAM.analytics.distance import compare

# --- hide
reference = BEAM.load_pretrained("foundation")
donor_a = BeamModel(reference.sample(50_000, seed=51))
# A shifted population: long CDR3s and the ...NEQFF ending are over-represented.
pool = reference.sample(400_000, seed=53)
shifted = [s for s in pool if len(s) >= 15 or s.endswith("NEQFF")][:50_000]
donor_b = BeamModel(shifted)
# --- show
# donor_a: as above; donor_b: a repertoire with longer CDR3s and more ...NEQFF endings
result = compare(donor_a, donor_b, top_k=200)
print(f"distance: {result.distance:.4f}")
print(f"from lengths: {result.length_squared_distance / result.squared_distance:.1%}"
      f"   from local patterns: {result.pattern_squared_distance / result.squared_distance:.1%}")

print("\nlocal patterns that differ most (probability given the length):")
print("length  position  pattern   donor A   donor B")
for p in result.patterns[:6]:
    print(f"{p.length:6}  {p.start:8}  {p.word:8}  {p.conditional_p:8.4f}  {p.conditional_q:8.4f}")

print("\noverlapping patterns grouped:")
for g in result.group_overlaps().groups[:4]:
    print(f"  length {g.length}, positions {g.start}-{g.end - 1}: {g.consensus}  "
          f"(more common in donor {'A' if g.direction == 'p' else 'B'})")
