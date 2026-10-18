from BEAM import BeamModel
# --- hide
import BEAM
reference = BEAM.load_pretrained("human_tcrb_cdr3_aa_foundation")
a = BeamModel(reference.sample(20_000, seed=31))
b = BeamModel(reference.sample(20_000, seed=32))
# --- show
# a and b: models built from two donors' repertoires
pooled = a.merged(b)                        # pool the counts: one bigger repertoire

s = "CASSLGQGYEQYF"
mixed = 0.5 * a.pseq(s) + 0.5 * b.pseq(s)   # mix the probabilities: pick a donor, then a cell
print(f"donor A {a.pseq(s):.3e}   donor B {b.pseq(s):.3e}")
print(f"pooled  {pooled.pseq(s):.3e}   mixed   {mixed:.3e}")
