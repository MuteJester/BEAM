"""Pruning, the frontier, and the per-length live cache.
"""
from __future__ import annotations

import importlib.util
import math
import random

import numpy as np
import pytest

from BEAM import BeamModel

from _helpers import AA, J_MOTIFS, make_corpus

# Pruning renormalises the length prior
# ═══════════════════════════════════════════════════════════════



# ═══════════════════════════════════════════════════════════════
# Pruning frontier
# ═══════════════════════════════════════════════════════════════



# ═══════════════════════════════════════════════════════════════
# Live structure: the frontier's edge masks and the per-length cache
# ═══════════════════════════════════════════════════════════════

def _enumerate_live_trellis(m, length):
    """Nodes and edges of the trellis at one length, counted from OUTSIDE.

    Independent of everything `topology` does. It enumerates every string of
    the requested length, asks the model which ones it can spell at all, and
    then reads the DAG structure straight off the survivors: a live node is a
    (layer, state) pair some spellable string passes through, and a live edge
    is a (layer, state, residue) triple some spellable string uses.

    That is the definition of "live" -- forward-reachable AND able to reach a
    valid ending -- restated as a property of complete paths, which is the one
    form that cannot be derived from the sweep it is checking. Only usable for
    lengths small enough to enumerate: 20**4 is 160,000 strings.
    """
    import itertools
    every = ["".join(t) for t in itertools.product(AA, repeat=length)]
    lp = m.log_pseq_batch(every)
    nodes, edges = set(), set()
    # The state holds exactly the residues one context needs, so its order is
    # the model's own. Read from the model rather than assumed: hard-coding it
    # is what made this oracle agree with a state width it was not checking.
    K = int(m.config["order"])
    for seq, v in zip(every, lp):
        if not math.isfinite(v):
            continue
        for i in range(length + 1):
            state = tuple(seq[i - K + j] if 0 <= i - K + j < i else "."
                          for j in range(K))
            nodes.add((i, state))
            if i < length:
                edges.add((i, state, seq[i]))
    by_layer = [0] * (length + 1)
    for i, _ in nodes:
        by_layer[i] += 1
    return len(nodes), len(edges), by_layer


@pytest.mark.parametrize("strict", [True, False])
def test_topology_matches_an_enumerated_trellis(strict):
    """`topology` counts what a brute-force enumeration of the support counts.

    This is the test that would catch the frontier's edge masks being wrong.
    `path_count` and `topology` no longer read a single edge WEIGHT: they read
    twenty bits per state that the frontier distilled from the weights while
    it was discovering reachability. A wrong bit adds or drops an edge, and
    nothing else in this file would notice, because every other check compares
    the sweeps against each other.
    """
    rng = random.Random(19)
    seqs = ["".join(rng.choice(AA) for _ in range(4)) for _ in range(120)]
    m = BeamModel(seqs, order=2, max_length=5, strict=strict)
    want_nodes, want_edges, want_layers = _enumerate_live_trellis(m, 4)
    t = m.topology(4)
    assert t["n_nodes"] == want_nodes
    assert t["n_edges"] == want_edges
    assert [int(v) for v in t["nodes_by_layer"]] == want_layers


@pytest.mark.parametrize("order", [2, 3, 4])
@pytest.mark.parametrize("strict", [True, False])
def test_live_structure_is_identical_on_every_repeat_call(order, strict):
    """Cached and uncached answers are the same double, not a close one.

    The per-length live structure is cached on the model, so the second call
    to any of these runs different code from the first. `==` rather than
    `approx` on purpose: a cache that shifts the last bit is a bug, not a
    tolerance.
    """
    seqs, abund = make_corpus(n=300, seed=5, lo=6, hi=9)
    m = BeamModel(seqs, abundances=abund, order=order, max_length=10, strict=strict)
    for length in m.observed_lengths:
        first = m.path_count(length)
        assert m.path_count(length) == first
        a = m.topology(length)
        b = m.topology(length)
        assert (a["n_nodes"], a["n_edges"]) == (b["n_nodes"], b["n_edges"])
        assert list(a["nodes_by_layer"]) == list(b["nodes_by_layer"])
    assert m.hill_number(0) == m.hill_number(0)


@pytest.mark.parametrize("order", [2, 3, 4])
def test_live_structure_is_rebuilt_after_every_mutation(order):
    """A mutated model answers as a model built that way from the start.

    The live structure is a projection of the finalized tables, so anything
    that changes the counts has to invalidate it. Comparing against a FRESH
    build rather than merely against "something different" is what makes this
    a staleness test: a cache that was not dropped keeps returning the old
    model's structure, which is still a perfectly self-consistent answer.
    """
    seqs, abund = make_corpus(n=200, seed=7, lo=6, hi=9)
    more, more_ab = make_corpus(n=120, seed=8, lo=6, hi=9)
    kw = dict(order=order, max_length=10, strict=True)
    length = 8

    base = BeamModel(seqs, abundances=abund, **kw)
    other = BeamModel(more, abundances=more_ab, **kw)
    both = BeamModel(seqs + more, abundances=abund + more_ab, **kw)

    warm = base.path_count(length)                     # fills the cache
    assert base.topology(length)["n_nodes"] > 0

    merged = base.merged(other)                        # copy + add
    assert merged.path_count(length) == both.path_count(length)
    assert (merged.topology(length)["n_nodes"]
            == both.topology(length)["n_nodes"])

    back = merged.copy()
    back.subtract(other)
    assert back.path_count(length) == warm
    assert back.topology(length)["n_nodes"] == base.topology(length)["n_nodes"]

    # posterior mixes in a prior; the point is only that the answer moves off
    # the cached one rather than being served from it.
    post = base.with_prior(other, kappa=50.0)
    assert post.path_count(length) > 0
    assert base.path_count(length) == warm, "the operand must be untouched"


def test_live_cache_does_not_survive_a_save_and_load(tmp_path):
    """A loaded model builds its own structure, and gets the same answer.

    The file stores counts, so loading re-finalizes. Nothing about the live
    structure is serialised and nothing should be: it is a projection of
    tables that the reader rebuilds.
    """
    seqs, abund = make_corpus(n=200, seed=13, lo=6, hi=9)
    m = BeamModel(seqs, abundances=abund, order=2, max_length=10, strict=True)
    want = {L: m.path_count(L) for L in m.observed_lengths}
    topo = {L: m.topology(L)["n_edges"] for L in m.observed_lengths}
    path = tmp_path / "live.edge"
    m.save(path)
    back = BeamModel.from_file(path)
    for L, v in want.items():
        assert back.path_count(L) == v
        assert back.topology(L)["n_edges"] == topo[L]


@pytest.mark.parametrize("threads", [1, 4, 8])
def test_live_structure_does_not_depend_on_the_thread_count(threads):
    """Thread count changes which thread builds which length, never the answer.

    The cache is filled from inside the parallel per-length loop, so this is
    also the test that two threads racing to publish the same length cannot
    produce two different structures.
    """
    import BEAM
    seqs, abund = make_corpus(n=400, seed=23, lo=8, hi=12)
    saved = BEAM.threads()
    try:
        BEAM.set_threads(1)
        m = BeamModel(seqs, abundances=abund, order=2, max_length=14, strict=True)
        want_pc = {L: m.path_count(L) for L in m.observed_lengths}
        want_topo = {L: m.topology(L)["n_edges"] for L in m.observed_lengths}
        want_d0 = m.hill_number(0)

        BEAM.set_threads(threads)
        m2 = BeamModel(seqs, abundances=abund, order=2, max_length=14, strict=True)
        assert m2.hill_number(0) == want_d0
        for L, v in want_pc.items():
            assert m2.path_count(L) == v
            assert m2.topology(L)["n_edges"] == want_topo[L]
    finally:
        BEAM.set_threads(saved)
