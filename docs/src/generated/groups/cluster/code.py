import itertools
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from BEAM import BeamModel
from BEAM.analytics.distance import compare
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
