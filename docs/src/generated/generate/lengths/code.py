from collections import Counter
import BEAM

model = BEAM.load_pretrained("human_tcrb_cdr3_aa_foundation")
draws = model.sample(50_000, seed=11)

observed = Counter(len(s) for s in draws)
expected = model.length_distribution
print("length  sampled share  model probability")
for L in range(10, 20):
    print(f"{L:6}  {observed[L] / len(draws):13.4f}  {expected[L]:17.4f}")

