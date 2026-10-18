from collections import Counter
import BEAM
from BEAM import BeamModel
from BEAM.analytics.distance import compare

r = compare(donor_a, donor_b, top_k=5000)

# The squared distance is the length part plus the pattern part
print(f"D^2 = {r.squared_distance:.6f}   (D = {r.distance:.4f})")
print(f"  length part  r.length_squared_distance   = {r.length_squared_distance:.6f}")
print(f"  pattern part r.pattern_squared_distance  = {r.pattern_squared_distance:.6f}")

# Each part, length by length: r.lengths
print(f"  sum of x.contribution over r.lengths         = {sum(x.contribution for x in r.lengths):.6f}")
print(f"  sum of x.pattern_contribution over r.lengths = {sum(x.pattern_contribution for x in r.lengths):.6f}")

# The pattern part, window by window: r.windows (one per length and start)
print(f"  sum of w.contribution over r.windows         = {sum(w.contribution for w in r.windows):.6f}")

# Word by word: r.patterns holds the strongest; r.remainder the rest
print(f"  listed words {r.reported_pattern_squared_distance:.6f} + remainder {r.remainder:.6f}")

# One word's term, by hand: sqrt(pi_A pi_B) (sqrt(p) - sqrt(q))^2 / (2 V_L)
w = r.patterns[0]
L = next(x for x in r.lengths if x.length == w.length)
V = Counter(x.length for x in r.windows)[w.length]          # windows at this length
by_hand = (L.p * L.q) ** 0.5 * (w.conditional_p ** 0.5 - w.conditional_q ** 0.5) ** 2 / (2 * V)
print(f"\nstrongest word: {w.word} at length {w.length}, positions {w.start}-{w.end - 1}")
print(f"  pi_A = {L.p:.4f}  pi_B = {L.q:.4f}  p = {w.conditional_p:.4f}  q = {w.conditional_q:.4f}  V = {V}")
print(f"  by hand {by_hand:.6f}   w.contribution {w.contribution:.6f}")
