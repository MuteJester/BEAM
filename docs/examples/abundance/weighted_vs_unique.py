import numpy as np
import BEAM
from BEAM import BeamModel
from _docs import figure  # hide
# --- hide
from _standins import population, expand
ref = BEAM.load_pretrained("human_tcrb_cdr3_aa_foundation")
seqs, counts = expand(population(ref, "A", 50_000, seed=7), seed=8, strength=2.5)
# --- show
# seqs: the distinct CDR3s of one repertoire; counts: how many cells carry each
by_cells = BeamModel(seqs, abundances=counts)    # each cell counts: describes the cells
by_sequence = BeamModel(seqs)                    # each distinct sequence counts once

top = seqs[int(np.argmax(counts))]
print(f"largest clone {top}: {max(counts) / sum(counts):.2%} of the cells")
print(f"P(largest clone)  weighted by cells: {by_cells.pseq(top):.2e}   unique: {by_sequence.pseq(top):.2e}")
print(f"distinct sequences: {len(seqs):,}   cells: {sum(counts):,}")
# --- hide
with figure("lengths", size=(6.4, 3.2)) as (fig, ax):
    Ls = list(range(8, 23))
    ax.plot(Ls, [by_cells.length_distribution.get(L, 0) for L in Ls], "o-", color="#ef785f", label="weighted by cells")
    ax.plot(Ls, [by_sequence.length_distribution.get(L, 0) for L in Ls], "s-", label="each sequence once")
    ax.set_xlabel("CDR3 length")
    ax.set_ylabel("probability of the length")
    ax.legend(frameon=False)
# --- show
