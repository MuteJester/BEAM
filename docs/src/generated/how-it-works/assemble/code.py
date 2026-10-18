import BEAM

model = BEAM.load_pretrained("human_tcrb_cdr3_aa_foundation")
s = "CASSLGQGYEQYF"
e = model.explain(s)

# ln P(s) = ln pi(L) + sum_i ln psi_i - ln Z(L), with ln psi_i = w_F ln q_F + w_B ln q_B
print(f"length term   ln pi(L)        = {e['log_length_prior']:9.4f}")
print(f"residue terms sum of ln psi_i = {e['combined'].sum():9.4f}")
print(f"normaliser    ln Z(L)         = {e['log_partition']:9.4f}")
print(f"ln P(s), assembled            = {e['log_length_prior'] + e['combined'].sum() - e['log_partition']:9.4f}")
print(f"ln P(s), from log_pseq        = {model.pseq(s, log=True):9.4f}")
