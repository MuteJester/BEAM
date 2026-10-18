"""Every level of the distance between two stand-in repertoires, for the anatomy interactive."""
from collections import defaultdict
import BEAM
from BEAM import BeamModel
from BEAM.analytics.distance import compare
from _docs import save_data

reference = BEAM.load_pretrained("human_tcrb_cdr3_aa_foundation")
donor_a = BeamModel(reference.sample(50_000, seed=51))
pool = reference.sample(400_000, seed=53)
donor_b = BeamModel([s for s in pool if len(s) >= 15 or s.endswith("NEQFF")][:50_000])

r = compare(donor_a, donor_b, top_k=20_000)
keep = {x.length for x in r.lengths if x.p + x.q > 1e-4}        # lengths that carry probability
n_windows = defaultdict(int)
for w in r.windows:
    n_windows[w.length] += 1
words = defaultdict(list)
for p in r.patterns:                                           # strongest first
    if p.length in keep and len(words[(p.length, p.start)]) < 8:
        words[(p.length, p.start)].append(p)
save_data("distance_anatomy", {
    "distance": r.distance, "squared": r.squared_distance,
    "length_part": r.length_squared_distance, "pattern_part": r.pattern_squared_distance,
    "reported": r.reported_pattern_squared_distance, "remainder": r.remainder, "width": r.width,
    "lengths": [{"length": x.length, "p": x.p, "q": x.q, "length_term": x.contribution,
                 "pattern_term": x.pattern_contribution, "windows": n_windows[x.length]}
                for x in r.lengths if x.length in keep],
    "windows": [{"length": w.length, "start": w.start, "contribution": w.contribution}
                for w in r.windows if w.length in keep],
    "words": [{"length": p.length, "start": p.start, "word": p.word, "p": p.conditional_p,
               "q": p.conditional_q, "contribution": p.contribution}
              for ws in words.values() for p in ws],
})
print(f"distance {r.distance:.4f}: {len(keep)} lengths, {sum(1 for w in r.windows if w.length in keep)} windows, "
      f"{sum(map(len, words.values()))} words kept")
