import math
import numpy as np
import BEAM

model = BEAM.load_pretrained("human_tcrb_cdr3_aa_foundation")
e = model.explain("CASSLGQGYEQYF")

bits = lambda log_p: -np.asarray(log_p) / math.log(2)          # surprisal, in bits
print("  i  residue   from start   from end   combined (psi_i)")
for i, (res, f, b, c) in enumerate(zip(e["residues"], bits(e["forward"]),
                                       bits(e["backward"]), bits(e["combined"])), start=1):
    print(f"{i:3}  {res:^7}  {f:11.2f}  {b:9.2f}  {c:17.2f}")

# ln P(s) = ln pi(L) - ln Z(L) + sum_i ln psi_i
total = e["log_length_prior"] - e["log_partition"] + e["combined"].sum()
print(f"\nln pi(L) = {e['log_length_prior']:.3f}   ln Z(L) = {e['log_partition']:.3f}   "
      f"sum of ln psi_i = {e['combined'].sum():.3f}")
print(f"ln pi(L) - ln Z(L) + sum of ln psi_i = {total:.6f}")
print(f"log_pseq                             = {e['log_pseq']:.6f}")

