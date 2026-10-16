"""Per-residue contributions of both readings, for the two-anchors animation."""
import math
import BEAM
from _docs import save_data

model = BEAM.load_pretrained("foundation")
w_start, w_end = model.config["weights"]
out = {"weights": [w_start, w_end], "sequences": []}
for s in ["CASSLGQGYEQYF", "CASSPRDRGLAGGYNEQFF", "CSARDRTGNGYTF", "CASSLGQGYEQYW"]:
    e = model.explain(s)
    out["sequences"].append({
        "sequence": s,
        "start_bits": [-v / math.log(2) for v in e["forward"]],
        "end_bits": [-v / math.log(2) for v in e["backward"]],
        "log_length": model.log_length_prior(len(s)),
        "log_partition": model.log_partition(len(s)),
        "log_p": model.pseq(s, log=True),
    })
save_data("two_anchors", out)
print(f"{len(out['sequences'])} sequences")
