"""The final law's conditionals, not locally normalised expert potentials.

Small nucleotide supports are enumerated independently through ``log_pseq``.
Masked queries separately check the streaming block-marginal construction;
neither reference reads the new transition kernel. Both alphabets include
unobserved contexts, and the amino-acid check uses the shipped order four.
"""
from __future__ import annotations

import itertools

import numpy as np
import pytest

import BEAM
from BEAM import BeamModel, BudgetExceeded
from BEAM.model import alphabets
from BEAM.query import BeamQuery


NT_SEQUENCES = ["A", "AT", "AN", "ACA", "GNN", "TCA", "ACGN",
                "NCGT", "TGNN", "TTGA", "NCGTA"]
NT_ABUNDANCES = [3, 11, 2, 17, 5, 23, 7, 13, 19, 29, 31]


def _nt_model(*, order=2, weights=(0.8, 0.35), sparse=True):
    return BeamModel(NT_SEQUENCES, abundances=NT_ABUNDANCES,
                     alphabet="nt", order=order, max_length=5,
                     weights=weights, sparse=sparse, alpha=2.5,
                     alpha_length=3.0)


@pytest.fixture(scope="module")
def nt_model():
    return _nt_model()


def _enumerated_law(model, length):
    symbols = alphabets.symbols(model.alphabet)
    digits = np.asarray(list(itertools.product(range(len(symbols)),
                                              repeat=length)), dtype=int)
    sequences = ["".join(symbols[d] for d in row) for row in digits]
    log_prob = model.log_pseq_batch(sequences) - model.log_length_prior(length)
    prob = np.exp(log_prob)
    assert prob.sum() == pytest.approx(1.0, abs=2e-13)
    return digits, log_prob, prob


def _context_indices(digits, position, order, n_residues):
    """The published row convention: oldest to newest, base R, no sentinel."""
    rows = np.zeros(len(digits), dtype=int)
    for i in range(max(0, position - order), position):
        rows = n_residues * rows + digits[:, i]
    return rows


@pytest.mark.parametrize("order,weights,sparse,lengths", [
    (2, (0.8, 0.35), True, (1, 2, 4, 5)),
    (2, (0.0, 1.0), False, (1, 3, 5)),
    (4, (0.25, 1.2), True, (1, 2, 3)),
])
def test_every_row_matches_the_enumerated_final_law(
        order, weights, sparse, lengths):
    """Includes boundary padding, the delayed backward sink and short L < k.

    A backward-only model is intentionally among the cases: treating the
    forward expert, or a locally normalised edge, as the conditional cannot
    accidentally pass that case.
    """
    model = _nt_model(order=order, weights=weights, sparse=sparse)
    R = model.n_residues
    for length in lengths:
        digits, expected_log_prob, prob = _enumerated_law(model, length)
        reconstructed = np.zeros(len(digits))
        for position in range(length):
            n_rows = R ** min(order, position)
            got = model.transition_rows(length, position)
            log_got = model.transition_rows(length, position, log=True)
            assert got.shape == log_got.shape == (n_rows, R)
            assert got.dtype == log_got.dtype == np.dtype(np.float64)
            assert np.isfinite(got).all() and (got > 0.0).all()
            assert np.isfinite(log_got).all()
            np.testing.assert_allclose(got.sum(axis=1), 1.0,
                                       rtol=0.0, atol=3e-14)
            np.testing.assert_allclose(np.exp(log_got), got,
                                       rtol=3e-13, atol=0.0)

            rows = _context_indices(digits, position, order, R)
            joint = np.bincount(rows * R + digits[:, position], weights=prob,
                                minlength=n_rows * R).reshape(n_rows, R)
            expected = joint / joint.sum(axis=1, keepdims=True)
            np.testing.assert_allclose(got, expected, rtol=3e-12, atol=3e-14)
            reconstructed += log_got[rows, digits[:, position]]
        np.testing.assert_allclose(reconstructed, expected_log_prob,
                                   rtol=0.0, atol=5e-12)


def test_streaming_blocks_match_independent_masked_queries(nt_model):
    """Keep only one context distribution and recover every 3-block at L=5.

    Marginalising the oldest residue moves the streaming distribution to the
    next window. The reference asks for each complete word as a mask event,
    so normalised rows alone cannot make an incorrect kernel pass this test.
    """
    model, length = nt_model, 5
    q = BeamQuery(model)
    symbols = alphabets.symbols(model.alphabet)
    R, order = len(symbols), model.config["order"]
    beta = np.ones(1)
    for position in range(length):
        block = beta[:, None] * model.transition_rows(length, position)
        width = min(order, position) + 1
        if position >= order:
            start = position - order
            expected = [q.probability(q.condition(length).run(start, "".join(w)))
                        for w in itertools.product(symbols, repeat=width)]
            np.testing.assert_allclose(block.ravel(), expected,
                                       rtol=3e-12, atol=3e-14)
            assert block.sum() == pytest.approx(1.0, abs=3e-14)
            # Lexicographic flattening makes the first axis the oldest symbol.
            beta = block.reshape(R, -1).sum(axis=0)
        else:
            beta = block.ravel()


@pytest.fixture(scope="module")
def aa_order_four():
    return BeamModel(["CASSYF", "CASSAF", "CASRYF", "CAGGWF", "WYAAC",
                      "CAYF"], abundances=[31, 17, 11, 5, 3, 2],
                     order=4, max_length=6, weights=(0.7, 0.45),
                     alpha=3.0, alpha_length=2.0)


@pytest.mark.parametrize("position,context", [(2, "CA"), (5, "WYAC")])
def test_shipped_order_aa_rows_match_conditioned_mask_sweeps(
        aa_order_four, position, context):
    """Prefix and full-context rows catch base-20/base-21 and reversal errors.

    The terminal row additionally needs the final backward sink; the model's
    full 20**4 context table is read even though training has only six strings.
    """
    model, length = aa_order_four, 6
    symbols = alphabets.symbols(model.alphabet)
    row = 0
    for residue in context:
        row = len(symbols) * row + symbols.index(residue)
    rows = model.transition_rows(length, position)
    assert rows.shape == (20 ** min(4, position), 20)
    query = BeamQuery(model)
    cond = query.condition(length).run(position - len(context), context)
    expected = query.marginals(cond)[position]
    np.testing.assert_allclose(rows[row], expected, rtol=4e-12, atol=3e-14)


@pytest.mark.parametrize("length,position,field", [
    (0, 0, "length"), (-1, 0, "length"), (6, 0, "length"),
    (2 ** 100, 0, "length"), (np.uint64(2 ** 64 - 1), 0, "length"),
    (5, -1, "position"), (5, 5, "position"), (5, 2 ** 100, "position"),
    (5, np.uint64(2 ** 64 - 1), "position"),
])
def test_invalid_indices_are_refused_before_narrowing(
        nt_model, length, position, field):
    with pytest.raises(ValueError, match=field):
        nt_model.transition_rows(length, position)


@pytest.mark.parametrize("bad", [2.0, 2.5, "2", None, float("nan"),
                                float("inf")])
@pytest.mark.parametrize("field", ["length", "position"])
def test_noninteger_indices_are_not_truncated(nt_model, bad, field):
    args = {"length": 5, "position": 2}
    args[field] = bad
    with pytest.raises(TypeError):
        nt_model.transition_rows(**args)


def test_numpy_integer_indices_and_returned_array_ownership(nt_model):
    expected = nt_model.transition_rows(5, 2)
    result = nt_model.transition_rows(np.int64(5), np.int32(2))
    np.testing.assert_array_equal(result, expected)
    result.fill(0.0)
    np.testing.assert_array_equal(nt_model.transition_rows(5, 2), expected)


def test_budget_refusal_is_diagnosable_and_does_not_change_the_model():
    model = _nt_model(order=4)
    expected = model.transition_rows(5, 4)
    original_budget = BEAM.memory_budget()
    try:
        BEAM.set_memory_budget(2048)
        with pytest.raises(BudgetExceeded, match="memory budget") as exc:
            model.transition_rows(5, 4)
        assert "set_memory_budget" in str(exc.value)
    finally:
        BEAM.set_memory_budget(original_budget)
    np.testing.assert_array_equal(model.transition_rows(5, 4), expected)
