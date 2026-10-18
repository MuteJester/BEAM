from BEAM import BeamModel, Mixture
# --- hide
import BEAM
from _standins import population
ref = BEAM.load_pretrained("human_tcrb_cdr3_aa_foundation")
def group_cells(kind, seeds):
    return [s for seed in seeds for s in population(ref, kind, 20_000, seed=seed)]
group_a = BeamModel(group_cells("A", (51, 52)))
group_a2 = BeamModel(group_cells("A", (53, 54)))
group_b = BeamModel(group_cells("B", (55, 56)))
group_c = BeamModel(group_cells("C", (57, 58)))
# --- show
# group_a, group_a2: two groups from the same population; group_b, group_c: two others
same = Mixture([group_a, group_a2])
different = Mixture([group_a, group_b, group_c])

# beta diversity: the effective number of distinct groups, from 1 (identical) to the count
print(f"two groups, one population:      beta = {same.beta_diversity(2.0).beta:.3f}")
print(f"three groups, three populations: beta = {different.beta_diversity(2.0).beta:.3f}")
