"""A layered-graph sweep small enough to draw: a nucleotide model with k = 2, length 8."""
import itertools
import numpy as np
import BEAM
from BEAM import BeamModel
from _docs import save_data

# Ten short DNA sequences of different lengths, so that, as in real data, the forward expert does
# not know where a sequence will end and the backward expert does.
train = ["TGTGCCAG", "TGTGCCAGCTT", "TGTGCTAG", "TGCGCCAGTT", "TGTGCAG", "TGTGCCAGAG", "TGTCCAG",
         "TGTGCCAGCAG", "TGCCAG", "TGTGCAAGTTT"]
model = BeamModel(train, alphabet="nt", order=2, max_length="data")
A, k, L = "ACGTN", 2, 8          # the five nucleotide residues, N for an unread base
wf, wb = model.config["weights"]
PAD = "·"

def factors(s):
    """ln of each expert's weighted factor at every residue of s."""
    e = model.explain(s)
    return wf * e["forward"], wb * e["backward"]

# The forward factor of residue j needs only the k residues before it, and the backward factor
# of residue m only the k residues after it; so one explain call per case gives each.
def fwd(j, ctx, x):                    # residue j (1-based) is x, after ctx (padded at the start)
    s = ("A" * L)[: j - 1 - len(ctx.strip(PAD))] + ctx.strip(PAD) + x
    return factors(s + "A" * (L - len(s)))[0][j - 1]

def bwd(m, x, after):                  # residue m is x, followed by after (padded at the end)
    s = "A" * (m - 1) + x + after.strip(PAD)
    return factors(s + "A" * (L - len(s)))[1][m - 1]

# The sweep: a state is the last k residues written. Appending residue x at position j multiplies
# in the forward factor of residue j and the backward factor of residue j - k, which the new
# residue completes; the backward factors of the last k residues are applied at the end.
layers = [{"states": [PAD * k], "W": [1.0]}]
edges = []
for j in range(1, L + 1):
    prev = layers[-1]
    nxt, W, E = {}, [], []
    for a, st in enumerate(prev["states"]):
        for x in A:
            ln = fwd(j, st, x)
            if j > k:
                ln += bwd(j - k, st[0], st[1:] + x)
            to = st[1:] + x
            if to not in nxt:
                nxt[to] = len(nxt)
                W.append(0.0)
            W[nxt[to]] += prev["W"][a] * np.exp(ln)
            E.append([a, nxt[to], np.exp(ln)])
    layers.append({"states": list(nxt), "W": W})
    edges.append(E)
terminal = [np.exp(bwd(L - 1, st[0], st[1] + PAD) + bwd(L, st[1], PAD * k)) for st in layers[-1]["states"]]
Z = sum(w * t for w, t in zip(layers[-1]["W"], terminal))

# Listing every sequence instead: all 5^8 of them, scored one by one.
every = ["".join(p) for p in itertools.product(A, repeat=L)]
total_p = np.exp(model.log_pseq_batch(every)).sum()
listed = total_p * np.exp(model.log_partition(L) - model.log_length_prior(L))
print(f"{len(every):,} sequences of length {L}; the sweep keeps at most {max(len(l['states']) for l in layers)} states "
      f"per position and walks {sum(len(e) for e in edges) + len(terminal):,} edges")
print(f"Z({L}) from the sweep:          {Z:.12f}")
print(f"Z({L}) from listing everything: {listed:.12f}")
print(f"log_partition({L}):             {np.exp(model.log_partition(L)):.12f}")

ref = BEAM.load_pretrained("human_tcrb_cdr3_aa_foundation")
top = ref.topology(13)
save_data("sweep_toy", {"alphabet": A, "k": k, "length": L, "train": train, "layers": layers,
                        "edges": edges, "terminal": terminal, "Z": Z, "listed": listed,
                        "n_sequences": len(every),
                        "reference": {"length": 13, "n_nodes": top["n_nodes"], "n_edges": top["n_edges"]}})
