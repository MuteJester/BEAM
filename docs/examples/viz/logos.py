"""Sequence logos of the reference model, by length, and common vs rare."""
import BEAM
from _docs import save_data

BEAM.set_memory_budget(16 << 30)
model = BEAM.load_pretrained("human_tcrb_cdr3_aa_foundation")
pack = lambda g: {"bits": [[round(float(v), 4) for v in row] for row in g.height_bits],  # noqa: E731
                  "consensus": g.consensus_sequence, "fraction": float(g.fraction),
                  "information": round(float(g.information_bits.sum()), 3)}
out = {"residues": model.insights.logo(15).residues, "all": {}, "common": {}, "rare": {}}
for L in range(10, 21):
    out["all"][L] = pack(model.insights.logo(L))
for L in range(13, 17):
    out["common"][L] = pack(model.insights.logo(L, pseq_above=1e-7))
    out["rare"][L] = pack(model.insights.logo(L, pseq_below=1e-7))
save_data("logos", out)
print("logos for lengths 10-20; common and rare for 13-16")
