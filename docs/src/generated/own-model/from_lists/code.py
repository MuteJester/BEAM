from collections import Counter
import BEAM
from BEAM import BeamModel

# sequences: list of CDR3 strings; abundances: how many cells carry each
model = BeamModel(sequences, abundances=abundances)

print(model)
print(f"distinct sequences: {len(sequences):,}   cells: {sum(abundances):,}")
print(f"P(CASSLGQGYEQYF) = {model.pseq('CASSLGQGYEQYF'):.2e}")
