import BEAM
from BEAM import PublicnessModel

model = BEAM.load_pretrained("human_tcrb_cdr3_aa_foundation")

# 20 people sampled to 10,000 cells, 50 to 100,000 and 30 to 1,000,000
depths = [10_000] * 20 + [100_000] * 50 + [1_000_000] * 30
cohort = PublicnessModel(depths)

mean, variance = cohort.moments(model.pseq("CASSLGQGAYNEQFF"))
print(f"{cohort.n_repertoires} people, expected carriers {mean:.1f}")
