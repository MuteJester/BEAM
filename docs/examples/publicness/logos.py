import BEAM
from _docs import figure, draw_logo  # hide

BEAM.set_memory_budget(16 << 30)          # these logos need about 6 GB
model = BEAM.load_pretrained("foundation")

common = model.insights.logo(15, pseq_above=1e-7)   # length-15 sequences with P > 1e-7
rare = model.insights.logo(15, pseq_below=1e-7)     # and the rest
print(f"common: {common.fraction:.1%} of the probability at length 15, consensus {common.consensus_sequence}")
print(f"rare:   {rare.fraction:.1%} of the probability at length 15, consensus {rare.consensus_sequence}")
print(f"information, common vs rare: {common.information_bits.sum():.1f} vs {rare.information_bits.sum():.1f} bits")
# --- hide
with figure("logos", size=(6.4, 4.4), nrows=2) as (fig, axes):
    for ax, logo, title in zip(axes, (common, rare), ("common (P > 1e-7)", "rare (P <= 1e-7)")):
        draw_logo(ax, logo.height_bits, logo.residues)
        ax.set_title(title, fontsize=10, loc="left")
# --- show
