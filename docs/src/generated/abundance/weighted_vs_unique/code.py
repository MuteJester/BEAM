import numpy as np
import BEAM
from BEAM import BeamModel
# seqs: the distinct CDR3s of one repertoire; counts: how many cells carry each
by_cells = BeamModel(seqs, abundances=counts)    # each cell counts: describes the cells
by_sequence = BeamModel(seqs)                    # each distinct sequence counts once

top = seqs[int(np.argmax(counts))]
print(f"largest clone {top}: {max(counts) / sum(counts):.2%} of the cells")
print(f"P(largest clone)  weighted by cells: {by_cells.pseq(top):.2e}   unique: {by_sequence.pseq(top):.2e}")
print(f"distinct sequences: {len(seqs):,}   cells: {sum(counts):,}")
