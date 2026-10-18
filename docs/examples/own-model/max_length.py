from collections import Counter
import BEAM
from BEAM import BeamModel

# --- hide
# The stand-in repertoire, plus two sequences of 31 residues.
reference = BEAM.load_pretrained("human_tcrb_cdr3_aa_foundation")
counts = Counter(reference.sample(50_000, seed=21))
counts["CASSLGQGYEQYFCASSLGQGYEQYFCASSL"] = 1
counts["CASSPGTGELFFCSARDRTGNGYTFCASSQE"] = 1
sequences, abundances = list(counts), list(counts.values())
# --- show
# sequences include two of 31 residues, longer than the default maximum of 27
default = BeamModel(sequences, abundances=abundances)                    # up to 27 residues
fitted = BeamModel(sequences, abundances=abundances, max_length="data")  # the longest given
chosen = BeamModel(sequences, abundances=abundances, max_length=40)      # a number you choose

for name, m in [("default", default), ('"data"', fitted), ("40", chosen)]:
    print(f"max_length={name:9}  represents up to {m.config['max_length']} residues,"
          f" skipped {m.n_skipped}")
