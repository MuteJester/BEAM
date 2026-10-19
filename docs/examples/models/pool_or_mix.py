import numpy as np
from BEAM import BeamModel, Mixture
# --- hide
import BEAM
ref = BEAM.load_pretrained("human_tcrb_cdr3_aa_foundation")
pool = ref.sample(2_000_000, seed=61)
group_a = BeamModel([s for s in pool if s.startswith("CASS") and s.endswith("YEQYF")][:20_000])
group_b = BeamModel([s for s in pool if s.startswith("CSA") and s.endswith("TEAFF")][:20_000])
# --- show
# group_a: 20,000 CDR3s that start CASS and end YEQYF
# group_b: 20,000 CDR3s that start CSA and end TEAFF
merged = group_a.merged(group_b)         # the counts added: one model
mixture = Mixture([group_a, group_b])    # pick a group (1/2 each), then a sequence from it

print("sequence           group A   group B   mixture    merged")
for s in ["CASSLGQGYEQYF", "CSARGNTEAFF", "CASSLGQGTEAFF"]:
    print(f"{s:16}  " + "  ".join(f"{m.pseq(s):8.1e}" for m in (group_a, group_b, mixture, merged)))

# How much of the merged model's probability falls on sequences that start like one group
# and end like the other, summed over every length
def share(model, start, end):
    total = 0.0
    for L, pi in model.length_distribution.items():
        if L >= len(start) + len(end):
            clamp = {**dict(enumerate(start)), **{L - len(end) + i: x for i, x in enumerate(end)}}
            total += pi * np.exp(model.log_clamp_probability(L, clamp))
    return total
mixed = share(merged, "CASS", "TEAFF") + share(merged, "CSA", "YEQYF")
print(f"\nmerged model: {mixed:.1%} of its probability starts like one group and ends like the other")

# Diversity: Hill number of order 2, the inverse chance that two draws are the same sequence
print(f"D2: group A {group_a.hill_number(2):,.0f}, group B {group_b.hill_number(2):,.0f}, "
      f"mixture {mixture.hill_number(2):,.0f}, merged {merged.hill_number(2):,.0f}")
beta = mixture.beta_diversity(2.0)
print(f"beta diversity of the mixture: {beta.beta:.2f} (gamma {beta.gamma:,.0f} / alpha {beta.alpha:,.0f})")
