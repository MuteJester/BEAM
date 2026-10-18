import math
import BEAM

model = BEAM.load_pretrained("human_tcrb_cdr3_aa_foundation")
s = "CASSLGQGYEQYF"
L = len(s)

e = model.explain(s)
w_start, w_end = model.config["weights"]
residues = sum(w_start * f + w_end * b for f, b in zip(e["forward"], e["backward"]))

log_p = model.log_length_prior(L) + residues - model.log_partition(L)
print(f"length term   ln pi(L)  = {model.log_length_prior(L):9.4f}")
print(f"residue terms           = {residues:9.4f}")
print(f"normaliser    ln Z(L)   = {model.log_partition(L):9.4f}")
print(f"ln P(s), assembled      = {log_p:9.4f}")
print(f"ln P(s), from log_pseq  = {model.pseq(s, log=True):9.4f}")
