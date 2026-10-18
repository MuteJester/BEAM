import math
import BEAM
from BEAM import PublicnessModel

model = BEAM.load_pretrained("human_tcrb_cdr3_aa_foundation")
cohort = PublicnessModel([50_000] * 100)      # 100 people, 50,000 cells sampled from each

for cdr3 in ["CASSLGQGYEQYF", "CASSLGQGAYNEQFF", "CASSPRDRGLAGGYNEQFF"]:
    mean, variance = cohort.moments(model.pseq(cdr3))
    print(f"{cdr3:20} expected carriers {mean:6.2f} ± {math.sqrt(variance):.2f}")
