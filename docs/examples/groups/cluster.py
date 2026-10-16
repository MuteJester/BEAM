import itertools
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from BEAM import BeamModel
from BEAM.analytics.distance import compare
from _docs import figure, save_data  # hide
# --- hide
import BEAM
from _standins import population
ref = BEAM.load_pretrained("foundation")
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
print("clusters:", " ".join(f"{g}{c}" for g, c in zip(labels, clusters)))

nearest = [labels[int(np.argsort(D[i])[1])] for i in range(n)]
print(f"nearest neighbour from the same group: {np.mean([a == b for a, b in zip(labels, nearest)]):.0%}")

same = [D[i, j] for i, j in itertools.combinations(range(n), 2) if labels[i] == labels[j]]
diff = [D[i, j] for i, j in itertools.combinations(range(n), 2) if labels[i] != labels[j]]
print(f"mean distance within groups: {np.mean(same):.4f}   between groups: {np.mean(diff):.4f}")
# --- hide
with figure("matrix", size=(5.2, 4.4)) as (fig, ax):
    im = ax.imshow(D, cmap="viridis")
    ax.set_xticks(range(n), [f"{g}{k % 4 + 1}" for k, g in enumerate(labels)], rotation=90)
    ax.set_yticks(range(n), [f"{g}{k % 4 + 1}" for k, g in enumerate(labels)])
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="distance")
J = np.eye(n) - 1.0 / n
B = -0.5 * J @ (D ** 2) @ J
w, v = np.linalg.eigh(B)
xy = v[:, -2:][:, ::-1] * np.sqrt(np.maximum(w[-2:][::-1], 0))
save_data("map", {"labels": labels, "names": [f"{g}{k % 4 + 1}" for k, g in enumerate(labels)],
                  "distance": D, "xy": xy, "clusters": clusters})
# --- show
