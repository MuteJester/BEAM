"""Each expert's full prediction at every position of one sequence, for the expert explainer."""
import numpy as np
import BEAM
from _docs import save_data

model = BEAM.load_pretrained("human_tcrb_cdr3_aa_foundation")
AA = "ACDEFGHIKLMNPQRSTVWY"
s = "CASSLGQGYEQYF"

# The forward expert's prediction at position i depends only on i and the residues before it,
# and the backward expert's only on the residues after it; so putting each of the 20 residues
# at i and reading explain gives each expert's whole prediction there.
forward = np.zeros((len(s), len(AA)))
backward = np.zeros((len(s), len(AA)))
for i in range(len(s)):
    for a, x in enumerate(AA):
        e = model.explain(s[:i] + x + s[i + 1:])
        forward[i, a] = np.exp(e["forward"][i])
        backward[i, a] = np.exp(e["backward"][i])
print(f"each prediction sums to 1 within {max(abs(forward.sum(1) - 1).max(), abs(backward.sum(1) - 1).max()):.1e}")

save_data("expert_walk", {"sequence": s, "alphabet": AA, "k": model.config["order"],
                          "weights": list(model.config["weights"]),
                          "forward": forward, "backward": backward})
i = s.index("L")
print(f"position {i + 1} ({s[i]}): forward expert gives it {forward[i, AA.index(s[i])]:.3f}, "
      f"backward expert {backward[i, AA.index(s[i])]:.3f}")
