"""The distance between two stand-in repertoires, split by length and pattern."""
import BEAM
from BEAM import BeamModel
from BEAM.analytics.distance import compare
from _docs import save_data

reference = BEAM.load_pretrained("foundation")
donor_a = BeamModel(reference.sample(50_000, seed=51))
pool = reference.sample(400_000, seed=53)
donor_b = BeamModel([s for s in pool if len(s) >= 15 or s.endswith("NEQFF")][:50_000])

r = compare(donor_a, donor_b, top_k=400)
save_data("distance_breakdown", {
    "squared_distance": r.squared_distance,
    "length_squared": r.length_squared_distance,
    "pattern_squared": r.pattern_squared_distance,
    "lengths": [{"length": x.length, "p": x.p, "q": x.q, "length_term": x.contribution,
                 "pattern_term": x.pattern_contribution} for x in r.lengths],
    "patterns": [{"length": p.length, "start": p.start, "word": p.word, "p": p.conditional_p,
                  "q": p.conditional_q, "contribution": p.contribution} for p in r.patterns],
})
print(f"distance {r.distance:.4f}; {len(r.lengths)} lengths, {len(r.patterns)} patterns kept")
