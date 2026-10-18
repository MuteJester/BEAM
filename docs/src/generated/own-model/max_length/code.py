from collections import Counter
import BEAM
from BEAM import BeamModel

# sequences include two of 31 residues, longer than the default maximum of 27
default = BeamModel(sequences, abundances=abundances)                    # up to 27 residues
fitted = BeamModel(sequences, abundances=abundances, max_length="data")  # the longest given
chosen = BeamModel(sequences, abundances=abundances, max_length=40)      # a number you choose

for name, m in [("default", default), ('"data"', fitted), ("40", chosen)]:
    print(f"max_length={name:9}  represents up to {m.config['max_length']} residues,"
          f" skipped {m.n_skipped}")
