from collections import Counter
import numpy as np
import BEAM
from BEAM.analytics.exact_occupancy import ExactOccupancy

# --- hide
from _standins import population, expand
reference = BEAM.load_pretrained("human_tcrb_cdr3_aa_foundation")
rng = np.random.default_rng(4)
# An ordinary sample: 100,000 independent cells from the reference population.
ordinary = population(reference, "A", 100_000, seed=31)
# A donor with expanded clones; a sample of 100,000 of its cells.
clones, sizes = expand(population(reference, "A", 300_000, seed=32), seed=33, strength=2.0)
share = np.asarray(sizes, float) / sum(sizes)
picked = rng.multinomial(100_000, share)
expanded = [clones[i] for i in np.nonzero(picked)[0] for _ in range(picked[i])]
# --- show
# ordinary, expanded: two samples of 100,000 cells (lists of sequences, one per cell)
model = BEAM.load_pretrained("human_tcrb_cdr3_aa_foundation")
occupancy = ExactOccupancy(model)
n = 100_000

def spectrum(cells):                  # how many sequences were seen exactly j times
    return Counter(Counter(cells).values())

seen = {"ordinary": spectrum(ordinary), "expanded": spectrum(expanded)}
print(" times seen   expected   ordinary sample   expanded sample")
for j in [1, 2, 3, 5, 10]:
    print(f"{j:10}  {occupancy.frequency(n, j).value:9,.0f}  {seen['ordinary'][j]:16,}"
          f"  {seen['expanded'][j]:16,}")
over = lambda s: sum(c for j, c in s.items() if j > 10)
expected_over = occupancy.richness(n).value - sum(occupancy.frequency(n, j).value for j in range(1, 11))
print(f"{'> 10':>10}  {expected_over:9,.0f}  {over(seen['ordinary']):16,}  {over(seen['expanded']):16,}")
