import itertools
import math
from BEAM import BeamModel
# --- hide
import BEAM
reference = BEAM.load_pretrained("foundation")
sequences = [s for s in reference.sample(20_000, seed=61) if len(s) <= 4] or ["CAS"]
sequences += ["CASF", "CAW", "CSF", "CAF"]
# --- show
# A small model over sequences of up to 4 residues, so that every one of them
# can be listed: 20 + 20**2 + 20**3 + 20**4 = 168,420 sequences.
model = BeamModel(sequences, order=2, max_length=4)

every = ["".join(t) for L in range(1, 5) for t in itertools.product("ACDEFGHIKLMNPQRSTVWY", repeat=L)]
total = math.fsum(math.exp(v) for v in model.log_pseq_batch(every) if v > -math.inf)
print(f"{len(every):,} sequences, total probability {total:.15f}")
