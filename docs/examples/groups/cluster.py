import itertools
import numpy as np
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import squareform
from BEAM import BeamModel
from BEAM.analytics.distance import compare
from _docs import figure, save_data  # hide
# --- hide
import BEAM
from _standins import population
ref = BEAM.load_pretrained("human_tcrb_cdr3_aa_foundation")
labels = [g for g in "ABC" for _ in range(4)]
repertoires = [population(ref, g, 50_000, seed=100 + i) for i, g in enumerate(labels)]
# --- show
# repertoires: 12 repertoires, 4 from each of groups A, B and C; labels: their groups
models = [BeamModel(cells, order=3) for cells in repertoires]   # order 3: fast for many pairs

n = len(models)
D = np.zeros((n, n))
for i, j in itertools.combinations(range(n), 2):
    D[i, j] = D[j, i] = compare(models[i], models[j]).distance

clusters = fcluster(linkage(squareform(D), "average"), 3, "maxclust")
print("cluster of each repertoire:", "  ".join(f"{g}{k % 4 + 1}:{c}" for k, (g, c) in enumerate(zip(labels, clusters))))

nearest = [labels[int(np.argsort(D[i])[1])] for i in range(n)]
print(f"nearest neighbour from the same group: {np.mean([a == b for a, b in zip(labels, nearest)]):.0%}")

same = [D[i, j] for i, j in itertools.combinations(range(n), 2) if labels[i] == labels[j]]
diff = [D[i, j] for i, j in itertools.combinations(range(n), 2) if labels[i] != labels[j]]
print(f"mean distance within groups: {np.mean(same):.4f}   between groups: {np.mean(diff):.4f}")
# --- hide
from matplotlib.patches import Rectangle
from _docs import PALETTE
# The clustering tree, built from the distances alone, and its cut into three clusters.
Z = linkage(squareform(D), "average")
order = dendrogram(Z, no_plot=True)["leaves"]
cut = (Z[-3, 2] + Z[-2, 2]) / 2
names = [f"{g}{k % 4 + 1}" for k, g in enumerate(labels)]
colours = {"A": PALETTE[0], "B": PALETTE[1], "C": PALETTE[2]}
with figure("matrix", size=(5.6, 5.8)) as (fig, ax):
    im = ax.imshow(D[np.ix_(order, order)], cmap="viridis")
    ax.set_xticks(range(n), [names[i] for i in order], rotation=90)
    ax.set_yticks(range(n), [names[i] for i in order])
    for k, i in enumerate(order):            # the group each repertoire came from
        ax.add_patch(Rectangle((k - 0.5, -1.45), 1, 0.7, color=colours[labels[i]], clip_on=False))
        ax.add_patch(Rectangle((-1.45, k - 0.5), 0.7, 1, color=colours[labels[i]], clip_on=False))
    top = ax.inset_axes([0, 1.09, 1, 0.34])
    dendrogram(Z, ax=top, no_labels=True, color_threshold=0, above_threshold_color="#7d7d7d")
    top.axhline(cut, ls="--", lw=1, color="#ef785f")
    top.text(0, cut, " cut into 3 clusters", va="bottom", fontsize=8, color="#ef785f")
    top.set_xlim(0, 10 * n)
    top.set_axis_off()
    fig.colorbar(im, ax=ax, label="distance", shrink=0.75)
save_data("map", {"labels": labels, "names": names, "distance": D, "clusters": clusters,
                  "linkage": Z, "order": order, "cut": cut})
# --- show
