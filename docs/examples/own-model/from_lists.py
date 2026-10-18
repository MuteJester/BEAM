from collections import Counter
import BEAM
from BEAM import BeamModel

# --- hide
# A stand-in repertoire: 50,000 cells drawn from the reference model.
reference = BEAM.load_pretrained("human_tcrb_cdr3_aa_foundation")
cells = reference.sample(50_000, seed=21)
counts = Counter(cells)
sequences, abundances = list(counts), list(counts.values())
# --- show
# sequences: list of CDR3 strings; abundances: how many cells carry each
model = BeamModel(sequences, abundances=abundances)

print(model)
print(f"distinct sequences: {len(sequences):,}   cells: {sum(abundances):,}")
print(f"P(CASSLGQGYEQYF) = {model.pseq('CASSLGQGYEQYF'):.2e}")
