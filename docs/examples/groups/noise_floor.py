import numpy as np
from BEAM import BeamModel
from BEAM.analytics.distance import compare
# --- hide
import BEAM
from _standins import population
ref = BEAM.load_pretrained("foundation")
donor_1 = population(ref, "A", 60_000, seed=121)
donor_2 = population(ref, "A", 60_000, seed=122)
donor_b = population(ref, "B", 60_000, seed=123)
# --- show
# donor_1, donor_2: two repertoires of one population; donor_b: one of another (stand-ins)
def model(cells):
    return BeamModel(cells, order=3)

def distance(x, y):
    return compare(model(x), model(y)).distance

half = 30_000
print(f"two halves of one repertoire:     {distance(donor_1[:half], donor_1[half:]):.4f}")
print(f"two repertoires, one population:  {distance(donor_1[:half], donor_2[:half]):.4f}")
print(f"two repertoires, two populations: {distance(donor_1[:half], donor_b[:half]):.4f}")

# unequal depth: the same comparison with one side ten times smaller
print(f"unequal depth (30,000 vs 3,000 cells), one population: {distance(donor_1[:half], donor_2[:3_000]):.4f}")
print(f"both subsampled to 3,000 cells, one population:         {distance(donor_1[:3_000], donor_2[:3_000]):.4f}")
print(f"both subsampled to 3,000 cells, two populations:        {distance(donor_1[:3_000], donor_b[:3_000]):.4f}")
