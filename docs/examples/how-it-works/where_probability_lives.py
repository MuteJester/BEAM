import numpy as np
import BEAM
from _docs import figure  # hide

BEAM.set_memory_budget(32 << 30)          # the fine probability grid below needs about 20 GB
model = BEAM.load_pretrained("human_tcrb_cdr3_aa_foundation")

# 1. By length: the model's probability against the share of all possible strings
strings = {L: 20.0 ** L for L in range(1, 28)}
total = sum(strings.values())
print("length   probability   share of possible strings")
for L in range(10, 21, 2):
    print(f"{L:6}   {model.length_distribution.get(L, 0):11.3g}   {strings[L] / total:25.2e}")
print(f"{27:6}   {model.length_distribution.get(27, 0):11.3g}   {strings[27] / total:25.2e}")

# 2. By probability: how much of the probability sits on how few sequences,
#    with every sequence grouped onto a fine grid of probabilities
p, w = model.pseq_analysis().atoms(16384).as_arrays()
order = np.argsort(p)[::-1]
mass = np.cumsum(p[order] * w[order])
support = np.cumsum(w[order]) / w.sum()
for cut in (1e-8, 1e-10, 1e-12):
    k = np.searchsorted(-p[order], -cut, side="right") - 1
    print(f"sequences with P >= {cut:.0e}: {mass[k]:.1%} of the probability, {support[k]:.1e} of the strings")
# --- hide
with figure("lengths", size=(6.4, 3.4)) as (fig, ax):
    Ls = list(range(5, 28))
    ax.semilogy(Ls, [model.length_distribution.get(L, 1e-12) for L in Ls], "o-", label="probability of the length")
    ax.semilogy(Ls, [strings[L] / total for L in Ls], "s--", label="share of possible strings")
    ax.set_ylim(1e-30, 2)
    ax.set_xlabel("CDR3 length")
    ax.legend(frameon=False)
# --- show
