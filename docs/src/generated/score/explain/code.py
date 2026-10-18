import math
import BEAM

model = BEAM.load_pretrained("human_tcrb_cdr3_aa_foundation")
e = model.explain("CASSLGQGYEQYF")

print("pos  residue  from start (bits)  from end (bits)")
for i, (res, n_logp, c_logp) in enumerate(zip(e["residues"], e["forward"], e["backward"])):
    print(f"{i:3}  {res:^7}  {-n_logp / math.log(2):17.2f}  {-c_logp / math.log(2):15.2f}")

