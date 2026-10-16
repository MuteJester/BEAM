import numpy as np
import BEAM
from BEAM import PublicnessModel

model = BEAM.load_pretrained("foundation")
cohort = PublicnessModel([50_000] * 100)

pmf = cohort.pmf(model.pseq("CASSLGQGAYNEQFF"))    # P(exactly k carriers), k = 0..100
print(f"P(nobody carries it)     {pmf[0]:.3f}")
print(f"P(1 to 5 people)         {pmf[1:6].sum():.3f}")
print(f"P(more than 5 people)    {pmf[6:].sum():.3f}")
print(f"most likely: {np.argmax(pmf)} people")

