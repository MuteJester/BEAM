import math
import BEAM

model = BEAM.load_pretrained("human_tcrb_cdr3_aa_foundation")

H = model.entropy()                       # Shannon entropy of the sequence distribution, nats
print(f"entropy H:                       {H:.3f} nats")
print(f"effective number of sequences:   D1 = exp(H) = {math.exp(H):.3e}")
print(f"inverse Simpson (Hill order 2):  D2 = {model.hill_number(2.0):.3e}")
