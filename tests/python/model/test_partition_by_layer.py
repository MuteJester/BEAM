"""`ln Z(L)` split across the positions it is a product over.

`ln P(s) = ln W(s) + ln pi(L) - ln Z(L)`. The last two terms belong to the
sequence's LENGTH, not to any position in it, so anything that reads a
per-position cell -- a K x L matrix over a panel of per-donor models, say --
has to place `-ln Z(L)` somewhere. Spreading it evenly over L is
arithmetically valid and analytically arbitrary.

It need not be arbitrary. `Z(L)` is a forward sum over the trellis and a
forward sum factorises by layer, so the split is the model's own. These tests
pin the two properties that makes it worth having: it is EXACT, and it is not
the uniform split in disguise.
"""
from __future__ import annotations

import numpy as np
import pytest

from BEAM import BeamModel

SEQS = ["CASSLGEQYF", "CASSQDTQYF", "CASSLGTEAFF", "CASSPGTGFF",
        "CASRDTQYF", "CASSLGEQYW", "CGSSLGEQYF"]
AB = [40.0, 25.0, 18.0, 15.0, 12.0, 10.0, 9.0]


@pytest.fixture(scope="module")
def model():
    return BeamModel(SEQS, abundances=AB, order=3, max_length=12)


def _live_lengths(m):
    return [L for L in m.observed_lengths if np.isfinite(m.log_partition(L))]


def test_the_last_entry_is_log_partition_bit_for_bit(model):
    """The telescoping end point, and the reason not to re-sum.

    Differencing and re-adding L terms accumulates about an ULP. The last
    running mass IS ln Z(L), so a caller wanting it exactly reads it there.
    """
    for L in _live_lengths(model):
        lm = model.log_partition_by_layer(L)
        assert lm.shape == (L,)
        assert lm[-1] == model.log_partition(L)


def test_the_shares_sum_back_to_log_z(model):
    for L in _live_lengths(model):
        lm = model.log_partition_by_layer(L)
        shares = np.diff(np.concatenate(([0.0], lm)))
        assert shares.sum() == pytest.approx(model.log_partition(L), abs=1e-12)


def test_the_split_is_not_the_uniform_one(model):
    """The point of the exercise.

    If every layer carried ln Z(L)/L, the arbitrary split this replaces would
    have been just as good and none of this would be worth its cost. On this
    fixture one layer carries more than half of -ln Z while others carry
    essentially none.
    """
    worst = 0.0
    for L in _live_lengths(model):
        z = model.log_partition(L)
        if abs(z) < 1e-9:
            continue
        shares = np.diff(np.concatenate(([0.0], model.log_partition_by_layer(L))))
        worst = max(worst, abs(shares).max() / (abs(z) / L))
    assert worst > 4.0, (
        f"the largest share is only {worst:.2f}x the uniform one, so the "
        f"split carries no information the uniform split lacks")


def test_it_refuses_what_it_cannot_answer(model):
    with pytest.raises(Exception):
        model.log_partition_by_layer(0)
    with pytest.raises(Exception):
        model.log_partition_by_layer(10_000)


def test_it_agrees_with_a_second_model_of_the_same_corpus(model):
    """Deterministic: the split is a property of the model, not of the call."""
    twin = BeamModel(SEQS, abundances=AB, order=3, max_length=12)
    for L in _live_lengths(model):
        assert np.array_equal(model.log_partition_by_layer(L),
                              twin.log_partition_by_layer(L))


# ── the per-position vector whose sum is the score ─────────────────────────

def test_the_vector_sums_to_log_pseq(model):
    """The defining property: stack K of these and the row sums ARE the K
    models' scores for that sequence."""
    for s in SEQS + ["CASSLGEQYW", "CGSSLGEQYF", "WWWWWWWW"]:
        v = model.log_pseq_by_position(s)
        assert v.shape == (len(s),)
        assert np.isfinite(v).all(), s
        assert v.sum() == pytest.approx(model.log_pseq(s), abs=1e-12)


def test_stacking_k_models_gives_the_matrix(model):
    """What the whole thing is for: K x L, row sums = each model's score.

    Each panel member sees five of the seven sequences, so a member may never
    have seen the one scored. Under the prior that is a finite row like any
    other, which is what lets the matrix be analysed without a -inf cell to
    special-case.
    """
    rng = np.random.default_rng(3)
    panel = []
    for k in range(4):
        idx = rng.choice(len(SEQS), size=5, replace=False)
        panel.append(BeamModel([SEQS[i] for i in idx],
                               abundances=[AB[i] for i in idx],
                               order=3, max_length=12))
    s = "CASSLGEQYF"
    M = np.vstack([m.log_pseq_by_position(s) for m in panel])
    assert M.shape == (4, len(s))
    assert np.isfinite(M).all()
    for k, m in enumerate(panel):
        assert M[k].sum() == pytest.approx(m.log_pseq(s), abs=1e-12)


def test_an_unseen_residue_is_finite_and_charged_where_it_sits():
    """A residue the corpus never emitted at a place is priced by the prior,
    not killed, and the vector says WHERE the price is paid.

    The length is unchanged, so Z(L) and its split are too, and the vector
    moves only through the per-residue terms: most at the substituted
    position, a little at the positions whose order-3 context reads it, and
    not at all anywhere else. Nothing may come back as nan.
    """
    m = BeamModel(["CASSLGEQYF", "CASSLGEQYW"], abundances=[3.0, 1.0],
                  order=3, max_length=12)
    base = "CASSLGEQYF"
    seen = m.log_pseq_by_position(base)
    for p in (9, 4):                 # a terminal residue, and a central one
        s = base[:p] + "Q" + base[p + 1:]
        v = m.log_pseq_by_position(s)
        assert np.isfinite(v).all(), s
        assert v.sum() == pytest.approx(m.log_pseq(s), abs=1e-12)
        d = v - seen
        assert int(np.argmin(d)) == p, (s, d)
        far = [i for i in range(len(s)) if abs(i - p) > 3]
        assert np.array_equal(d[far], np.zeros(len(far))), (s, d)


def test_a_length_the_model_never_saw(model):
    """A length the corpus never had still has a share of the length prior
    and a partition function, so its breakdown is shared out and sums to its
    score like any other rather than reporting a zero."""
    short = "CAS"
    assert len(short) not in {len(s) for s in SEQS}
    v = model.log_pseq_by_position(short)
    assert v.shape == (3,)
    assert np.isfinite(v).all()
    assert v.sum() == pytest.approx(model.log_pseq(short), abs=1e-12)


def test_an_empty_sequence_is_refused(model):
    with pytest.raises(Exception):
        model.log_pseq_by_position("")
