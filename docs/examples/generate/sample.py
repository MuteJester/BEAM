import BEAM

model = BEAM.load_pretrained("human_tcrb_cdr3_aa_foundation")

print(model.sample(5, seed=1))             # any length
print(model.sample(5, length=12, seed=1))  # exactly 12 residues
