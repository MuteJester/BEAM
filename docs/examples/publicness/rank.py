import numpy as np
import BEAM
from BEAM import PublicnessModel

model = BEAM.load_pretrained("human_tcrb_cdr3_aa_foundation")
# --- hide
from collections import Counter
repertoire = list(Counter(model.sample(20_000, seed=41)))
# --- show
# repertoire: the distinct CDR3s of one sample
cohort = PublicnessModel([100_000] * 100)

probabilities = np.exp(model.log_pseq_batch(repertoire))
expected, _ = cohort.moments(probabilities)          # one call for the whole list

order = np.argsort(expected)[::-1]
print("most public sequences of this sample (expected carriers out of 100):")
for i in order[:5]:
    print(f"  {repertoire[i]:18} {expected[i]:5.1f}")
print(f"expected in fewer than one person: {np.mean(expected < 1):.0%} of its sequences")
