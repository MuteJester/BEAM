from BEAM import BeamModel, Mixture
# group_a, group_a2: two groups from the same population; group_b, group_c: two others
same = Mixture([group_a, group_a2])
different = Mixture([group_a, group_b, group_c])

# beta diversity: the effective number of distinct groups, from 1 (identical) to the count
print(f"two groups, one population:      beta = {same.beta_diversity(2.0).beta:.3f}")
print(f"three groups, three populations: beta = {different.beta_diversity(2.0).beta:.3f}")
