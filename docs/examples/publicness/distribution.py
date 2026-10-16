import numpy as np
import BEAM
from BEAM import PublicnessModel
from _docs import figure  # hide

model = BEAM.load_pretrained("foundation")
cohort = PublicnessModel([50_000] * 100)

pmf = cohort.pmf(model.pseq("CASSLGQGAYNEQFF"))    # P(exactly k carriers), k = 0..100
print(f"P(nobody carries it)     {pmf[0]:.3f}")
print(f"P(1 to 5 people)         {pmf[1:6].sum():.3f}")
print(f"P(more than 5 people)    {pmf[6:].sum():.3f}")
print(f"most likely: {np.argmax(pmf)} people")

# --- hide
with figure("carriers", size=(6.4, 3.4)) as (fig, ax):
    for cdr3 in ["CASSLGQGYEQYF", "CASSLGQGAYNEQFF", "CASSPRDRGLAGGYNEQFF"]:
        p = cohort.pmf(model.pseq(cdr3))
        k = np.arange(len(p))
        keep = p > 1e-4
        ax.plot(k[keep], p[keep], "o-", ms=3, lw=1.2, label=cdr3)
    ax.set_xlabel("number of people (out of 100) whose sample contains the sequence")
    ax.set_ylabel("probability")
    ax.legend(frameon=False, fontsize=8)
# --- show
