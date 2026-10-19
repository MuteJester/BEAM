"""The hierarchical prior on a model of 20,000 cells: each level written out, and held-out tests."""
from collections import Counter, defaultdict
import numpy as np
import BEAM
from BEAM import BeamModel
from _docs import save_data

ref = BEAM.load_pretrained("human_tcrb_cdr3_aa_foundation")
cells = list(ref.sample(25_000, seed=7))
train, test = cells[:20_000], cells[20_000:]
model = BeamModel(train)
alpha = list(model.config["alpha"])
k = len(alpha) - 1

# Each expert's counts: a row per position from its own end and context of j residues before it
# in its reading direction ('^' past the end of the sequence), for every level j = 0..k.
rows = [defaultdict(Counter), defaultdict(Counter)]
for s in train:
    for e, r in enumerate((s, s[::-1])):
        for i, x in enumerate(r):
            for j in range(k + 1):
                rows[e][(i, r[max(0, i - j):i].rjust(j, "^"))][x] += 1

def ladder(sequence, i, backward):
    """Level by level, the counts behind one expert's probability of residue i."""
    r = sequence[::-1] if backward else sequence
    p = i if not backward else len(sequence) - 1 - i
    x, levels, q = r[p], [], 1 / 20
    for j in range(k + 1):
        ctx = r[max(0, p - j):p].rjust(j, "^")
        row = rows[backward][(p, ctx)]
        n, nx = sum(row.values()), row[x]
        q = (nx + alpha[j] * q) / (n + alpha[j])
        shown = ctx[::-1] if backward else ctx          # in sequence order
        levels.append({"context": shown.replace("^", "·"), "n": n, "nx": nx})
    return levels, q

seen = set(train)
unseen = next(s for s in test if s not in seen and len(s) == 14)
common = Counter(s for s in train if len(s) == 13).most_common(1)[0][0]
probes = {"seen in training": common, "new": unseen, "implausible": "CASSWWWWWWEQYF"}
ladders, worst = {}, 0.0
for name, s in probes.items():
    e = model.explain(s)
    per = []
    for i in range(len(s)):
        f_levels, qf = ladder(s, i, False)
        b_levels, qb = ladder(s, i, True)
        worst = max(worst, abs(np.log(qf) - e["forward"][i]), abs(np.log(qb) - e["backward"][i]))
        per.append({"forward": f_levels, "backward": b_levels})
    ladders[s] = per
print(f"strengths fitted from the counts: " + ", ".join(f"{a:,.1f}" for a in alpha))
print(f"the ladder written out agrees with BEAM's experts to {worst:.1e}")

# Held out: how likely are 5,000 new cells when every strength is multiplied by m?
mults = np.logspace(-2, 2, 33)
held_out = []
for m in mults:
    scaled = BeamModel(train, alpha=[a * m for a in alpha], alpha_length=model.config["alpha_length"])
    held_out.append(float(np.mean(scaled.log_pseq_batch(test))))
best = int(np.argmax(held_out))
at_one = held_out[16]
print(f"{sum(s not in seen for s in test) / len(test):.0%} of the held-out cells are sequences the model never saw")
print(f"mean ln P of a held-out cell: {at_one:.3f} at BEAM's strengths, {held_out[best]:.3f} at the best "
      f"multiple (x{mults[best]:.2f}); {held_out[0]:.3f} at x0.01, {held_out[-1]:.3f} at x100")

save_data("prior_strength", {"alpha": alpha, "probes": probes, "ladders": ladders,
                             "multipliers": mults, "held_out": held_out,
                             "n_train": len(train), "n_test": len(test),
                             "unseen_share": sum(s not in seen for s in test) / len(test),
                             "reference_alpha": list(ref.config["alpha"])})
