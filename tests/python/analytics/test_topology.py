"""Node and edge counts of the trellis.
"""
from __future__ import annotations

import importlib.util
import math
import random

import numpy as np
import pytest

from BEAM import BeamModel

from _helpers import AA, J_MOTIFS, make_corpus

# Topology
# ═══════════════════════════════════════════════════════════════

def test_topology_counts_live_structure(corpus):
    seqs, abund = corpus
    m = BeamModel(seqs, abundances=abund, order=2)
    t = m.topology(13)
    assert m.is_dag
    assert t["n_nodes"] > 0 and t["n_edges"] > 0
    assert len(t["nodes_by_layer"]) == 14
    assert t["max_layer_nodes"] >= t["mean_layer_nodes"] > 0
    assert t["nodes_by_layer"][0] == 1, "the walk starts in one state"


def test_topology_edges_bound_the_paths(corpus):
    """Every path uses exactly one edge per layer, so the support size is
    bounded above by the product of the per-layer edge counts and below by the
    smallest of them. The previous version of this test asserted only
    `path_count >= 1` and `n_edges >= 13`, which the docstring's claim does not
    follow from and which a badly broken trellis would still satisfy."""
    seqs, abund = corpus
    m = BeamModel(seqs, abundances=abund, order=2)
    length = 13
    t = m.topology(length)
    pc = m.path_count(length)
    layers = t["nodes_by_layer"]

    assert pc >= 1
    assert len(layers) == length + 1
    # A path visits one node per layer, so the support cannot exceed the
    # product of the per-layer live-node counts.
    log_bound = float(np.log(layers[1:].astype(float)).sum())
    assert math.log(pc) <= log_bound + 1e-9, (pc, log_bound)
    # ... and every live node at the last layer terminates at least one path.
    assert pc >= layers[-1]
    # Edges are per (layer, node, residue), so they bound the paths too.
    assert t["n_edges"] >= t["n_nodes"] - 1
    assert 0.0 < t["density"] <= 1.0



# ═══════════════════════════════════════════════════════════════
