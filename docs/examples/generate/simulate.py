import BEAM

model = BEAM.load_pretrained("foundation")
draws = model.simulate(4, seed=3)

for cdr3, logp in zip(draws.sequences, draws.log_probs):
    print(f"{cdr3:20} log P = {logp:.3f}")
