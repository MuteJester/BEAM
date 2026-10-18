import BEAM

model = BEAM.load_pretrained("human_tcrb_cdr3_aa_foundation")
p_length = model.length_distribution          # {length: probability}

for cdr3 in ["CASSLGQGYEQYF", "CASSLGQGAYNEQFF", "CASSPRDRGLAGGYNEQFF"]:
    p = model.pseq(cdr3)
    L = len(cdr3)
    print(f"{cdr3:20} L={L}  P(s) = {p:.2e}   P(L) = {p_length[L]:.3f}"
          f"   P(s | L) = {p / p_length[L]:.2e}")
