"""Expected occupancy spectra of the reference population across depths, for the interactive."""
import numpy as np
import BEAM
from BEAM.analytics.exact_occupancy import ExactOccupancy
from _docs import save_data

model = BEAM.load_pretrained("human_tcrb_cdr3_aa_foundation")
occupancy = ExactOccupancy(model)
J = 20                                              # classes seen 1..J times, then "more"
depths = [int(round(d)) for d in np.logspace(3, 9, 13)]
rows = []
for n in depths:
    f = [occupancy.frequency(n, j) for j in range(1, J + 1)]
    r = occupancy.richness(n)
    counts = [v.value for v in f]
    # seen more than J times; at small depths these are rounding noise around 0
    more = max(0.0, r.value - sum(counts))
    cells_more = max(0.0, n - sum(j * c for j, c in enumerate(counts, start=1)))
    rows.append({"depth": n, "distinct": r.value, "counts": counts, "more": more,
                 "cells_more": cells_more})
save_data("occupancy_spectrum", {"J": J, "rows": rows})
for row in rows[::3]:
    print(f"{row['depth']:>13,} cells: {row['distinct']:>13,.0f} distinct, "
          f"{row['counts'][0] / row['distinct']:6.1%} seen once, {row['more']:>11,.0f} seen more than {J} times")
