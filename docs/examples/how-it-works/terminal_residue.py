import BEAM

model = BEAM.load_pretrained("human_tcrb_cdr3_aa_foundation")

real = "CASSLGQGYEQYF"
swapped = "CASSLGQGYEQYW"          # the same, with the final F changed to W

print(f"{real}   {model.pseq(real):.3e}")
print(f"{swapped}   {model.pseq(swapped):.3e}")
print(f"ratio: {model.pseq(real) / model.pseq(swapped):,.0f}")
