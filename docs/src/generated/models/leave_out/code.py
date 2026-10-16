import numpy as np
from BEAM import BeamModel
# corpus_cells: the repertoires a reference is built from; cohort_cells: your cohort,
# which is part of that corpus
reference = BeamModel(corpus_cells + cohort_cells)
cohort = BeamModel(cohort_cells)

held_out = reference.without(cohort)              # the reference with the cohort's counts removed
rebuilt = BeamModel(corpus_cells)                 # the same thing, built from scratch

cohort_seqs = sorted(set(cohort_cells))
difference = np.abs(held_out.log_pseq_batch(cohort_seqs) - rebuilt.log_pseq_batch(cohort_seqs)).max()
print(f"largest difference from rebuilding without the cohort: {difference:.1e}")
gain = np.mean(reference.log_pseq_batch(cohort_seqs) - held_out.log_pseq_batch(cohort_seqs))
print(f"mean log P of the cohort's sequences is {gain:.2f} nats higher when the cohort is included")
