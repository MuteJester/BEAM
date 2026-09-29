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

def _enumerate_live_trellis(m, length, alphabet):
    """Nodes and edges of the trellis at one length, counted from OUTSIDE.

    Independent of everything `topology` does. It enumerates every string of
    the requested length over the model's WHOLE alphabet, asks the model which
    ones it can spell at all, and then reads the DAG structure straight off the
    survivors: a live node is a (layer, state) pair some spellable string
    passes through, and a live edge is a (layer, state, residue) triple some
    spellable string uses.

    That is the definition of "live" -- forward-reachable AND able to reach a
    valid ending -- restated as a property of complete paths, which is the one
    form that cannot be derived from the sweep it is checking. Only usable for
    lengths small enough to enumerate: 5**6 is 15,625 nucleotide strings.
    """
    import itertools
    every = ["".join(t) for t in itertools.product(alphabet, repeat=length)]
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


def test_topology_matches_an_enumerated_trellis():
    """`topology` counts what a brute-force enumeration of the support counts.

    This is the test that would catch the frontier's edge masks being wrong.
    `path_count` and `topology` no longer read a single edge WEIGHT: they read
    one bit per residue per state that the frontier distilled from the weights
    while it was discovering reachability. A wrong bit adds or drops an edge,
    and nothing else in this file would notice, because every other check
    compares the sweeps against each other.

    Under the prior every string is spellable, so the enumerated trellis is
    the complete one; what is checked is that the masks say so, bit for bit.
    Nucleotides keep the whole length enumerable.
    """
    rng = random.Random(19)
    seqs = ["".join(rng.choice("ACGT") for _ in range(6)) for _ in range(120)]
    m = BeamModel(seqs, alphabet="nt", order=2, max_length=7)
    want_nodes, want_edges, want_layers = _enumerate_live_trellis(m, 6,
                                                                  "ACGTN")
    t = m.topology(6)
    assert t["n_nodes"] == want_nodes
    assert t["n_edges"] == want_edges
    assert [int(v) for v in t["nodes_by_layer"]] == want_layers


@pytest.mark.parametrize("order", [2, 3, 4])
def test_live_structure_is_identical_on_every_repeat_call(order):
    """Cached and uncached answers are the same double, not a close one.

    The per-length live structure is cached on the model, so the second call
    to any of these runs different code from the first. `==` rather than
    `approx` on purpose: a cache that shifts the last bit is a bug, not a
    tolerance.
    """
    seqs, abund = make_corpus(n=300, seed=5, lo=6, hi=9)
    m = BeamModel(seqs, abundances=abund, order=order, max_length=10)
    # The lengths the corpus has. Every length up to max_length is live now,
    # but the cache is per length and these four exercise it as well as ten.
    for length in sorted(set(map(len, seqs))):
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

    Under the prior the live structure is the complete trellis for every
    model, so structure alone can no longer tell a stale cache from a fresh
    one. Scores can: `log_pseq` reads the finalized weights and the cached
    partition function of the length, and integer counts re-finalize to the
    same bits, so they are compared exactly. (Scores rather than the entropy:
    an order-4 entropy is a sweep over every length, several seconds each,
    and a score reads the same tables for the price of a lookup.)
    """
    seqs, abund = make_corpus(n=200, seed=7, lo=6, hi=9)
    more, more_ab = make_corpus(n=120, seed=8, lo=6, hi=9)
    kw = dict(order=order, max_length=10)
    length = 8

    base = BeamModel(seqs, abundances=abund, **kw)
    other = BeamModel(more, abundances=more_ab, **kw)
    both = BeamModel(seqs + more, abundances=abund + more_ab, **kw)

    probe = [x for x in seqs + more if len(x) == length][:20]

    def scores(m):
        return list(m.log_pseq_batch(probe))

    warm = base.path_count(length)                     # fills the cache
    assert warm == 20.0 ** length
    assert base.topology(length)["n_nodes"] > 0
    s_base = scores(base)

    merged = base.merged(other)                        # copy + add
    assert merged.path_count(length) == both.path_count(length)
    assert (merged.topology(length)["n_nodes"]
            == both.topology(length)["n_nodes"])
    assert scores(merged) == scores(both)

    back = merged.copy()
    back.subtract(other)
    assert back.path_count(length) == warm
    assert back.topology(length)["n_nodes"] == base.topology(length)["n_nodes"]
    assert scores(back) == s_base

    # posterior mixes in a prior; the point is only that the answer moves off
    # the cached one rather than being served from it.
    post = base.with_prior(other, kappa=50.0)
    assert post.path_count(length) > 0
    assert scores(post) != s_base
    assert base.path_count(length) == warm, "the operand must be untouched"
    assert scores(base) == s_base, "the operand must be untouched"


def test_live_cache_does_not_survive_a_save_and_load(tmp_path):
    """A loaded model builds its own structure, and gets the same answer.

    The file stores counts, so loading re-finalizes. Nothing about the live
    structure is serialised and nothing should be: it is a projection of
    tables that the reader rebuilds.
    """
    seqs, abund = make_corpus(n=200, seed=13, lo=6, hi=9)
    m = BeamModel(seqs, abundances=abund, order=2, max_length=10)
    want = {L: m.path_count(L) for L in m.observed_lengths}
    topo = {L: m.topology(L)["n_edges"] for L in m.observed_lengths}
    path = tmp_path / "live.edge"
    m.save(path)
    back = BeamModel.from_file(path)
    for L, v in want.items():
        assert back.path_count(L) == v
        assert back.topology(L)["n_edges"] == topo[L]
    # The structure is complete either way, so the weights it was rebuilt
    # from are what distinguish a faithful reload: same counts, same prior,
    # same bits.
    assert back.entropy() == m.entropy()


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
        m = BeamModel(seqs, abundances=abund, order=2, max_length=14)
        want_pc = {L: m.path_count(L) for L in m.observed_lengths}
        want_topo = {L: m.topology(L)["n_edges"] for L in m.observed_lengths}
        want_d0 = m.hill_number(0)
        want_d2 = m.hill_number(2)

        BEAM.set_threads(threads)
        m2 = BeamModel(seqs, abundances=abund, order=2, max_length=14)
        assert m2.hill_number(0) == want_d0
        for L, v in want_pc.items():
            assert m2.path_count(L) == v
            assert m2.topology(L)["n_edges"] == want_topo[L]
        # The structure is complete at every thread count, so it cannot show
        # a race on its own; D2 reads the weights through the same parallel
        # per-length loop and must come out the same double.
        assert m2.hill_number(2) == want_d2
    finally:
        BEAM.set_threads(saved)
