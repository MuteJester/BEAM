import BEAM

model = BEAM.load_pretrained("human_tcrb_cdr3_aa_foundation")

# One sequence: a probability, or its natural logarithm.
print(model.pseq("CASSLGQGYEQYF"))
print(model.pseq("CASSLGQGYEQYF", log=True))

# Many sequences at once.
cdr3s = ["CASSLGQGYEQYF", "CASSPGTGELFF", "CSARDRTGNGYTF"]
for cdr3, logp in zip(cdr3s, model.log_pseq_batch(cdr3s)):
    print(f"{cdr3:16} log P = {logp:7.3f}")
