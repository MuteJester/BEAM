import BEAM

model = BEAM.load_pretrained("foundation")

for cdr3 in model.sample(5, seed=1):
    print(cdr3)
