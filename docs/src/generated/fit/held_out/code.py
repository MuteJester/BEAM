import numpy as np
from BEAM import BeamModel
# cells: one repertoire's cells. Build on 80% of them; keep 20% aside to check against.
train, test = cells[:64_000], cells[64_000:]
model = BeamModel(train)
AA = "ACDEFGHIKLMNPQRSTVWY"

# 1. Lengths: the model's length distribution against the held-out cells'
lengths = np.array([len(s) for s in test])
observed = {L: np.mean(lengths == L) for L in set(lengths)}
tv_length = 0.5 * sum(abs(model.length_distribution.get(L, 0) - observed.get(L, 0))
                      for L in set(observed) | set(model.length_distribution))
print(f"length distribution, total variation:   {tv_length:.3f}")

# 2. Composition at each position, for length 15
L = 15
seqs = [s for s in test if len(s) == L]
empirical = np.array([[np.mean([s[i] == a for s in seqs]) for a in AA] for i in range(L)])
marginals = model.positional_marginals(L)
print(f"positional composition (length {L}), mean total variation: "
      f"{0.5 * np.abs(marginals - empirical).sum(axis=1).mean():.3f}")

# 3. Held-out sequences the model never saw: BEAM against a model of independent
#    positions with the same length and positional composition
held_out = sorted(set(test) - set(train))
by_length = {L: np.log(model.positional_marginals(L)) for L in {len(s) for s in held_out}}

def independent(s):
    m = by_length[len(s)]
    return np.log(model.length_distribution[len(s)]) + sum(m[i, AA.index(a)] for i, a in enumerate(s))

beam = model.log_pseq_batch(held_out)
indep = np.array([independent(s) for s in held_out])
print(f"{len(held_out):,} unseen held-out sequences: BEAM scores them {np.mean(beam - indep):.2f} nats higher on average"
      f" ({np.mean(beam > indep):.0%} of them)")
