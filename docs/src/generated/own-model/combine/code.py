from BEAM import BeamModel
# a and b: models built from two donors' repertoires
pooled = a.merged(b)                        # pool the counts: one bigger repertoire

s = "CASSLGQGYEQYF"
mixed = 0.5 * a.pseq(s) + 0.5 * b.pseq(s)   # mix the probabilities: pick a donor, then a cell
print(f"donor A {a.pseq(s):.3e}   donor B {b.pseq(s):.3e}")
print(f"pooled  {pooled.pseq(s):.3e}   mixed   {mixed:.3e}")
