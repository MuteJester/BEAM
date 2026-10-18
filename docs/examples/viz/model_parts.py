"""Data for the animations of how BEAM works: counting, the prior, assembling, the sweep."""
from collections import defaultdict
import numpy as np
import BEAM
from BEAM import BeamModel
from _docs import save_data

# A small model, small enough to show its counts: six CDR3s, contexts of k = 2 residues.
train = ["CASSLGQGYEQYF", "CASSPGTGELFF", "CASSLGQGAYNEQFF", "CSARDRTGNGYTF", "CASSQETQYF", "CASSLGGNEQFF"]
probes = ["CASSLGQGYEQYW", "CASSPRDRGTEAFF"]
k = 2
toy = BeamModel(train, order=k)
alpha = list(toy.config["alpha"])

# The prior, written out: each expert counts residues by position from its own end and the k
# residues before it, then backs off level by level from the uniform 1/20 with strengths alpha.
def expert_q(sequence, backward):
    reads = [s[::-1] if backward else s for s in train]
    r = sequence[::-1] if backward else sequence
    out = []
    for i, x in enumerate(r):
        p = 1 / 20
        for j in range(k + 1):
            ctx = r[max(0, i - j):i].rjust(j, "^")
            row = defaultdict(float)
            for t in reads:
                if i < len(t) and t[max(0, i - j):i].rjust(j, "^") == ctx:
                    row[t[i]] += 1
            n = sum(row.values())
            p = (row[x] + alpha[j] * p) / (n + alpha[j])
        out.append(np.log(p))
    return np.array(out[::-1] if backward else out)

worst = 0.0
explained = {}
for s in train + probes:
    e = toy.explain(s)
    worst = max(worst, np.abs(expert_q(s, False) - e["forward"]).max(),
                np.abs(expert_q(s, True) - e["backward"]).max())
    explained[s] = {"forward": e["forward"], "backward": e["backward"]}
print(f"the prior written out agrees with BEAM's experts to {worst:.1e}")

# The reference model: one sequence's probability, assembled; the sweep behind Z(L).
ref = BEAM.load_pretrained("human_tcrb_cdr3_aa_foundation")
s = "CASSLGQGYEQYF"
e = ref.explain(s)
top = ref.topology(len(s))
layers = ref.log_partition_by_layer(len(s))
save_data("model_parts", {
    "toy": {"train": train, "probes": probes, "k": k, "alpha": alpha, "weights": list(toy.config["weights"]),
            "explained": explained},
    "assemble": {"sequence": s, "forward": e["forward"], "backward": e["backward"],
                 "combined": e["combined"], "log_length_prior": e["log_length_prior"],
                 "log_partition": e["log_partition"], "log_pseq": e["log_pseq"]},
    "sweep": {"length": len(s), "log_mass": layers, "nodes_by_layer": top["nodes_by_layer"],
              "n_nodes": top["n_nodes"], "n_edges": top["n_edges"], "log_partition": ref.log_partition(len(s))},
})
print(f"{s}: ln P = {e['log_pseq']:.4f}; length {len(s)}: {top['n_nodes']:,} nodes, {top['n_edges']:,} edges, "
      f"ln Z = {layers[-1]:.4f}")
