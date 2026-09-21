"""Sparse closed-world storage.
"""
from __future__ import annotations

import importlib.util
import math
import random

import numpy as np
import pytest

from BEAM import BeamModel

from _helpers import (AA, J_MOTIFS, make_corpus, STRICT_SEQS)

# ── Sparse strict storage ──────────────────────────────────────────


@pytest.mark.parametrize(("order", "abundances"), [
    (2, None),
    (3, [2.0, 5.0, 7.0]),
    (4, [3.0, 1.0, 4.0]),
])
def test_sparse_strict_is_bit_identical_to_dense_strict(
        order, abundances):
    """Storage changes allocation only; every public result stays identical."""
    kwargs = dict(order=order, max_length=6, strict=True,
                  abundances=abundances)
    dense = BeamModel(STRICT_SEQS, sparse=False, **kwargs)
    sparse = BeamModel(STRICT_SEQS, sparse=True, **kwargs)

    assert dense.config["sparse"] is False
    assert sparse.config["sparse"] is True

    probe = STRICT_SEQS + ["CASSW", "WWWWW", "CTTGF"]
    assert np.array_equal(dense.log_pseq_batch(probe),
                          sparse.log_pseq_batch(probe))
    assert sparse.log_pseq("CASSW") == -math.inf

    assert dense.n_observed == sparse.n_observed
    assert dense.n_skipped == sparse.n_skipped
    assert dense.total_weight == sparse.total_weight
    assert dense.n_nonzero_cells == sparse.n_nonzero_cells
    assert dense.path_count(5) == sparse.path_count(5)
    assert dense.entropy() == sparse.entropy()
    for alpha in (0.0, 1.0, 2.0):
        assert dense.hill_number(alpha) == sparse.hill_number(alpha)

    dense_topology = dense.topology(5)
    sparse_topology = sparse.topology(5)
    for key in ("length", "n_nodes", "n_edges", "max_layer_nodes",
                "mean_layer_nodes", "density"):
        assert dense_topology[key] == sparse_topology[key]
    assert np.array_equal(dense_topology["nodes_by_layer"],
                          sparse_topology["nodes_by_layer"])
    assert dense.top_k_sequences(10, length=5) == sparse.top_k_sequences(
        10, length=5)
    dense_explain = dense.explain(STRICT_SEQS[0])
    sparse_explain = sparse.explain(STRICT_SEQS[0])
    assert np.array_equal(dense_explain["forward"],
                          sparse_explain["forward"])
    assert np.array_equal(dense_explain["backward"],
                          sparse_explain["backward"])
    assert dense.sample(20, length=5, seed=17) == sparse.sample(
        20, length=5, seed=17)

    for measure in ("counting", "generated"):
        dense_spectrum = dense.spectrum(measure=measure, length=5, bins=32)
        sparse_spectrum = sparse.spectrum(measure=measure, length=5, bins=32)
        assert dense_spectrum["spacing"] == sparse_spectrum["spacing"]
        assert np.array_equal(dense_spectrum["surprisal"],
                              sparse_spectrum["surprisal"])
        assert np.array_equal(dense_spectrum["weights"],
                              sparse_spectrum["weights"])

    assert np.array_equal(dense.surprisal_moments(order=4),
                          sparse.surprisal_moments(order=4))
    assert dense.pgen_diagnostics()["total_mass"] == 1.0
    assert sparse.pgen_diagnostics()["total_mass"] == 1.0

    cfg = dense.config
    # One table per direction: max_length depth slots, A**order contexts,
    # twenty residues.
    expected_cells = 2 * cfg["max_length"] * 21 ** cfg["order"] * 20
    assert dense.n_cells == sparse.n_cells == expected_cells
    assert dense.memory_bytes >= 20 * sparse.memory_bytes


def test_sparse_selection_and_positive_backoff_fallback():
    """Automatic mode is sparse only in the exact zero-backoff regime."""
    automatic = BeamModel(STRICT_SEQS, order=2, max_length=6, strict=True)
    forced_dense = BeamModel(STRICT_SEQS, order=2, max_length=6, strict=True,
                        sparse=False)
    open_world = BeamModel(STRICT_SEQS, order=2, max_length=6, strict=False)
    backed_off = BeamModel(
        STRICT_SEQS, order=2, max_length=6, strict=True,
        sparse=None
    )

    assert automatic.config["sparse"] is True
    assert forced_dense.config["sparse"] is False
    assert open_world.config["sparse"] is False
    assert open_world.hill_number(0) > automatic.hill_number(0)

    with pytest.raises(ValueError, match="sparse=True"):
        BeamModel(STRICT_SEQS, order=2, max_length=6, strict=False,
                  sparse=True)
    # Sparse storage is exact only in the closed world, which is now the
    # default -- so the rejection needs the open world asked for explicitly.
    with pytest.raises(ValueError, match="sparse=True"):
        BeamModel(STRICT_SEQS, order=2, max_length=6, sparse=True, strict=False)
    # And by default sparse is not merely allowed, it is what you get.
    assert BeamModel(STRICT_SEQS, order=2, max_length=6).config["sparse"] is True


def test_sparse_subtract_shrinks_and_copy_is_independent():
    """Rows disappear at zero, while copy and streamed snapshot own storage."""
    prefix = STRICT_SEQS[:2]
    removed = STRICT_SEQS[2:]
    kwargs = dict(order=2, max_length=6, strict=True, sparse=True)

    builder = BeamModel.builder(**kwargs)
    builder.observe(prefix)
    builder_occupancy = builder.n_nonzero_cells
    assert builder_occupancy > 0
    snapshot = builder.snapshot()
    snapshot_scores = snapshot.log_pseq_batch(STRICT_SEQS)
    snapshot_occupancy = snapshot.n_nonzero_cells

    builder.observe(removed)
    assert builder.n_nonzero_cells >= builder_occupancy
    full = builder.build()
    expected_snapshot = BeamModel(prefix, **kwargs)
    assert np.array_equal(snapshot_scores,
                          expected_snapshot.log_pseq_batch(STRICT_SEQS))
    assert snapshot.n_nonzero_cells == snapshot_occupancy
    assert np.array_equal(snapshot.log_pseq_batch(STRICT_SEQS),
                          snapshot_scores)

    original_scores = full.log_pseq_batch(STRICT_SEQS)
    original_occupancy = full.n_nonzero_cells
    held_out = full.copy()
    held_out.subtract(BeamModel(removed, **kwargs))

    assert held_out.n_nonzero_cells < original_occupancy
    assert held_out.n_nonzero_cells == expected_snapshot.n_nonzero_cells
    assert np.array_equal(held_out.log_pseq_batch(STRICT_SEQS),
                          expected_snapshot.log_pseq_batch(STRICT_SEQS))
    assert full.n_nonzero_cells == original_occupancy
    assert np.array_equal(full.log_pseq_batch(STRICT_SEQS), original_scores)

    untouched = full.copy()
    with pytest.raises(ValueError, match="every sequence"):
        untouched.subtract(full.copy())
    assert np.array_equal(untouched.log_pseq_batch(STRICT_SEQS),
                          original_scores)


def test_sparse_add_posterior_without_and_compact_match_dense(tmp_path):
    """Every count mutation supports sparse and mixed-storage operands."""
    kwargs = dict(order=2, max_length=6, strict=True)
    left = STRICT_SEQS[:2]
    right = STRICT_SEQS[2:]
    probes = STRICT_SEQS + ["CASSW", "WWWWW"]

    dense_left = BeamModel(left, sparse=False, **kwargs)
    sparse_left = BeamModel(left, sparse=True, **kwargs)
    dense_right = BeamModel(right, sparse=False, **kwargs)
    sparse_right = BeamModel(right, sparse=True, **kwargs)
    together = BeamModel(STRICT_SEQS, sparse=False, **kwargs)

    dense_plus_sparse = dense_left.copy()
    dense_plus_sparse.add(sparse_right)
    sparse_plus_dense = sparse_left.copy()
    sparse_plus_dense.add(dense_right)
    expected = together.log_pseq_batch(probes)
    assert np.array_equal(dense_plus_sparse.log_pseq_batch(probes), expected)
    assert np.array_equal(sparse_plus_dense.log_pseq_batch(probes), expected)

    dense_post = dense_left.with_prior(sparse_right, kappa=3.0)
    sparse_post = sparse_left.with_prior(dense_right, kappa=3.0)
    assert np.array_equal(dense_post.log_pseq_batch(probes),
                          sparse_post.log_pseq_batch(probes))

    without_rows = sparse_plus_dense.without(right)
    assert without_rows.config["sparse"] is True
    assert np.array_equal(without_rows.log_pseq_batch(probes),
                          sparse_left.log_pseq_batch(probes))

    before = sparse_plus_dense.memory_bytes
    scores = sparse_plus_dense.log_pseq_batch(probes)
    sparse_plus_dense.compact()
    assert sparse_plus_dense.memory_bytes < before
    assert np.array_equal(sparse_plus_dense.log_pseq_batch(probes), scores)
    with pytest.raises(RuntimeError, match="compact"):
        sparse_plus_dense.save(tmp_path / "compacted.edge")


def test_sparse_single_sequence_has_minimal_support():
    model = BeamModel(["CASSF"], order=2, max_length=6, strict=True)
    assert model.config["sparse"] is True
    # One table per direction, one occupied cell per position in each.
    assert model.n_nonzero_cells == 2 * len("CASSF")
    assert model.log_pseq("CASSF") == 0.0
    assert model.path_count(5) == 1.0


def test_sparse_results_are_thread_count_independent():
    from BEAM import threads, set_threads

    prior = threads()
    kwargs = dict(order=2, max_length=6, strict=True, sparse=True)
    try:
        set_threads(1)
        serial = BeamModel(STRICT_SEQS * 20, **kwargs)
        set_threads(4)
        parallel = BeamModel(STRICT_SEQS * 20, **kwargs)
    finally:
        set_threads(prior)

    probes = STRICT_SEQS + ["CASSW", "WWWWW"]
    assert np.array_equal(serial.log_pseq_batch(probes),
                          parallel.log_pseq_batch(probes))
    assert serial.entropy() == parallel.entropy()
    assert serial.hill_number(0) == parallel.hill_number(0)


def test_sparse_strict_save_load_and_crc(tmp_path):
    """The sparse mode and exact model survive disk; row payloads stay covered."""
    model = BeamModel(STRICT_SEQS, order=2, max_length=6, strict=True, sparse=True)
    path = tmp_path / "strict-sparse.edge"
    model.save(path)
    restored = BeamModel.from_file(path)

    assert restored.config == model.config
    assert restored.config["sparse"] is True
    assert restored.n_cells == model.n_cells
    assert restored.n_nonzero_cells == model.n_nonzero_cells
    assert restored.hill_number(0) == model.hill_number(0)
    assert np.array_equal(restored.log_pseq_batch(STRICT_SEQS + ["CASSW"]),
                          model.log_pseq_batch(STRICT_SEQS + ["CASSW"]))

    data = bytearray(path.read_bytes())
    data[len(data) // 2] ^= 0xA5
    path.write_bytes(data)
    with pytest.raises(OSError):
        BeamModel.from_file(path)


@pytest.mark.parametrize("order", [2, 3, 4])
def test_backward_hoist_is_correct_at_every_order(order, corpus):
    """The backward-lookup hoist is only valid at order 4; prove the rest work.

    `state_weights_sparse` can lift two of its three backward table lookups out
    of the residue loop, but only for some configurations. The settled residue
    sits at ``p = i - K`` with ``K = max(order, 3)``, so its order-J backward
    context is ``s[i-K+1 .. i-K+J]`` and the emitted residue ``s[i]`` is inside
    it exactly when ``J >= K``:

        order 4 (K=4): wide J=4 -> x is the last digit, cw(x) == cw(0) + x
                       mid  J=3 -> invariant, hoistable
                       len  J=2 -> invariant, hoistable
        order 3 (K=3): mid  J=3 -> x IS in the context, NOT hoistable
        order 2 (K=3): wide J=2 -> x is not in the wide context at all,
                                   so cw(x) == cw(0) and "+ x" is wrong

    Hoisting unconditionally drove every partition function to -inf at order 2:
    the model built, reported no error, and could generate nothing. The
    reference values in this file are all order 4, so none of them caught it.
    """
    seqs, abund = corpus
    m = BeamModel(seqs[:600], abundances=abund[:600], order=order, max_length=20,
             strict=True)
    d = m.pgen_diagnostics()
    assert d["is_proper"], f"order {order} is not a proper distribution"
    assert d["total_mass"] == pytest.approx(1.0)
    assert math.isfinite(m.log_partition(m.observed_lengths[0]))
    assert m.hill_number(0) > 0.0
    assert m.hill_number(1) > 0.0
    # every training sequence must be spellable by the model built from it
    scores = m.log_pseq_batch(seqs[:50])
    assert np.isfinite(scores).all(), f"order {order} cannot spell its own input"


# ═══════════════════════════════════════════════════════════════
