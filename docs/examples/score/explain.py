import math
import BEAM
from _docs import figure  # hide

model = BEAM.load_pretrained("foundation")
e = model.explain("CASSLGQGYEQYF")

print("pos  residue  from start (bits)  from end (bits)")
for i, (res, n_logp, c_logp) in enumerate(zip(e["residues"], e["forward"], e["backward"])):
    print(f"{i:3}  {res:^7}  {-n_logp / math.log(2):17.2f}  {-c_logp / math.log(2):15.2f}")

# --- hide
with figure("explain", size=(6.4, 3.2)) as (fig, ax):
    pos = range(len(e["residues"]))
    n_bits = [-v / math.log(2) for v in e["forward"]]
    c_bits = [-v / math.log(2) for v in e["backward"]]
    ax.bar([p - 0.2 for p in pos], n_bits, width=0.4, label="read from the start")
    ax.bar([p + 0.2 for p in pos], c_bits, width=0.4, label="read from the end")
    ax.set_xticks(list(pos), list(e["residues"]))
    ax.set_ylabel("surprisal (bits)")
    ax.set_xlabel("residue of CASSLGQGYEQYF")
    ax.legend(frameon=False)
# --- show
