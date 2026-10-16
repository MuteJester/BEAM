from collections import Counter
import BEAM
from _docs import figure  # hide

model = BEAM.load_pretrained("foundation")
draws = model.sample(50_000, seed=11)

observed = Counter(len(s) for s in draws)
expected = model.length_distribution
print("length  sampled share  model probability")
for L in range(10, 20):
    print(f"{L:6}  {observed[L] / len(draws):13.4f}  {expected[L]:17.4f}")

# --- hide
with figure("lengths", size=(6.4, 3.2)) as (fig, ax):
    Ls = list(range(5, 28))
    ax.bar(Ls, [observed[L] / len(draws) for L in Ls], label="50,000 sampled sequences")
    ax.plot(Ls, [expected.get(L, 0.0) for L in Ls], "o-", color="#e07a3f", ms=3,
            label="model probability")
    ax.set_xlabel("CDR3 length (residues)")
    ax.set_ylabel("share")
    ax.legend(frameon=False)
# --- show
