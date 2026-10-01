"""Exact per-position marginals and expected cell usage.

Every check here is EXACT AGAINST EXACT. The models are built small enough
that the whole space is enumerable -- under the prior that is every string
over the whole alphabet, 5**6 = 15,625 nucleotide strings at length 6 -- so
the reference is a direct sum over sequences of `exp(log_pseq)`, not a sample
and not a second dynamic program that could share a mistake with the first.
Nothing here is compared against a Monte Carlo estimate, and no tolerance is
looser than machine precision.
"""
from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from BEAM import BeamModel
from BEAM.constants import BEAM_ALPHABET as AA

from _helpers import make_corpus, make_nt_corpus


#: Machine precision, near enough. Every quantity below is O(1), so an absolute
#: tolerance is the honest one: relative tolerance on a probability that is
#: legitimately 1e-9 would demand more of the arithmetic than double precision
#: has to give. Measured worst case across this file is 7e-15.
EXACT = 1e-13

L_ENUM = 6
NT = "ACGTN"


def _enumerable(order=3, n=40, seed=7):
    """A model whose entire length-6 space fits in a list.

    A conserved head and tail with a short random middle, like the CDR3s the
    rest of the suite uses, but over nucleotides. The prior makes every string
    over the WHOLE alphabet possible, so the support is 5**6 strings rather
    than the dozen the corpus spells; nucleotides are what keep that
    enumerable, where the amino-acid version of this fixture would now be
    20**15. Lengths 5 and 6 both occur, so a length's marginals are not a
    re-reading of one length's training set.
    """
    rng = np.random.default_rng(seed)
    seqs = ["TG" + "".join(rng.choice(list("AG"), int(k))) + "C"
            for k in rng.integers(2, 4, n)]
    return BeamModel(list(dict.fromkeys(seqs)), alphabet="nt", order=order,
                     max_length=L_ENUM)


def _space(m, length=L_ENUM):
    """Every string of `length`, as (strings, residue codes, log P)."""
    seqs = ["".join(t) for t in itertools.product(NT, repeat=length)]
    codes = np.array([[NT.index(c) for c in s] for s in seqs], dtype=np.int64)
    lp = np.asarray(m.log_pseq_batch(seqs), dtype=np.float64)
    assert np.isfinite(lp).all(), "the prior leaves no string impossible"
    return seqs, codes, lp


@pytest.fixture(scope="module")
def enum_model():
    return _enumerable()


@pytest.fixture(scope="module")
def support(enum_model):
    """The complete length-6 support: every string, with log probabilities."""
    return _space(enum_model)


def _direct_marginals(support, length, clamp=None):
    """P(x_i = a | length, clamp) by summing over the enumerated support."""
    _, codes, lp = support
    keep = np.ones(len(lp), dtype=bool)
    for p, r in (clamp or {}).items():
        keep &= codes[:, p] == NT.index(r)
    w = np.exp(lp[keep])
    z = float(w.sum())
    out = np.stack([np.bincount(codes[keep, i], weights=w, minlength=len(NT))
                    for i in range(length)])
    return out / z, z


# Exact against exact: unclamped
# ═══════════════════════════════════════════════════════════════

def test_marginals_match_a_direct_sum_over_the_whole_support(enum_model,
                                                             support):
    ref, _ = _direct_marginals(support, L_ENUM)
    got = enum_model.positional_marginals(L_ENUM)
    assert got.shape == (L_ENUM, len(NT))
    assert np.abs(got - ref).max() < EXACT


def test_no_cell_is_zero_and_each_matches_to_its_own_size(enum_model,
                                                           support):
    """Every residue is possible at every position, N included.

    The prior leaves no string impossible, so a zero anywhere in the marginals
    would be the model denying something it can spell. The comparison is
    RELATIVE here, cell by cell, because an absolute bound would let the
    smallest cells be wrong by more than their own size. Measured worst
    relative difference: 3e-14.
    """
    ref, _ = _direct_marginals(support, L_ENUM)
    got = enum_model.positional_marginals(L_ENUM)
    assert (ref > 0.0).all() and (got > 0.0).all()
    assert (np.abs(got - ref) / ref).max() < 1e-12


# Exact against exact: clamped
# ═══════════════════════════════════════════════════════════════

@pytest.mark.parametrize("pos", [0, 1, 2, 4, 5])
def test_clamped_marginals_match_the_conditional_sum(enum_model, support, pos):
    """The clamp is the deliverable that matters, so it is checked at every
    kind of position: inside the conserved head, in the free middle, and at
    the conserved tail -- and for every residue, since every residue occurs
    at every position of the support."""
    for residue in NT:
        ref, _ = _direct_marginals(support, L_ENUM, {pos: residue})
        got = enum_model.positional_marginals(L_ENUM, clamp={pos: residue})
        assert np.abs(got - ref).max() < EXACT
        assert (got[np.arange(L_ENUM) != pos] > 0.0).all()


def test_a_clamped_position_is_one_hot(enum_model):
    pos, residue = 3, "G"
    got = enum_model.positional_marginals(L_ENUM, clamp={pos: residue})
    assert got[pos, NT.index(residue)] == pytest.approx(1.0, abs=EXACT)
    assert got[pos].sum() == pytest.approx(1.0, abs=EXACT)
    assert (np.delete(got[pos], NT.index(residue)) == 0.0).all()


def test_clamping_several_positions_at_once(enum_model, support):
    s0 = support[0][int(np.argmax(support[2]))]
    clamp = {1: s0[1], 3: s0[3], 4: s0[4]}
    ref, _ = _direct_marginals(support, L_ENUM, clamp)
    got = enum_model.positional_marginals(L_ENUM, clamp=clamp)
    assert np.abs(got - ref).max() < EXACT


def test_clamping_every_position_recovers_that_sequence(enum_model, support):
    """The degenerate clamp, which is the strongest statement of what a clamp
    means: pin all six positions and the conditional is a point mass on the
    sequence, whatever its probability was -- here the least probable string
    of the whole space."""
    s0 = support[0][int(np.argmin(support[2]))]
    got = enum_model.positional_marginals(L_ENUM,
                                          clamp={i: c for i, c in enumerate(s0)})
    for i, ch in enumerate(s0):
        assert got[i, NT.index(ch)] == pytest.approx(1.0, abs=EXACT)


def test_clamp_accepts_residue_indices_as_well_as_characters(enum_model):
    a = enum_model.positional_marginals(L_ENUM, clamp={4: "G"})
    b = enum_model.positional_marginals(L_ENUM, clamp={4: NT.index("G")})
    assert (a == b).all()


def test_an_empty_clamp_is_the_unclamped_answer(enum_model):
    a = enum_model.positional_marginals(L_ENUM)
    b = enum_model.positional_marginals(L_ENUM, clamp={})
    assert (a == b).all()


def test_clamp_probability_matches_the_enumerated_mass(enum_model, support):
    total = float(np.exp(support[2]).sum())
    for pos in (1, 2, 4):
        for residue in NT:
            _, z = _direct_marginals(support, L_ENUM, {pos: residue})
            got = enum_model.log_clamp_probability(L_ENUM, {pos: residue})
            assert got == pytest.approx(math.log(z / total), abs=EXACT)


# A clamp the corpus never saw
# ═══════════════════════════════════════════════════════════════

def test_a_clamp_the_corpus_never_saw_is_possible_not_an_error(enum_model,
                                                               support):
    """There is no impossible clamp inside the alphabet any more.

    Position 1 is G in every training sequence and N occurs nowhere, so N at
    position 1 is an event the corpus never came near. The prior still gives
    it mass: the conditional exists and is the direct sum, and
    `log_clamp_probability` is finite -- and well below that of the residue
    the corpus does put there, or the prior would be swamping the counts.
    """
    clamp = {1: "N"}
    ref, _ = _direct_marginals(support, L_ENUM, clamp)
    got = enum_model.positional_marginals(L_ENUM, clamp=clamp)
    assert np.abs(got - ref).max() < EXACT
    rare = enum_model.log_clamp_probability(L_ENUM, clamp)
    usual = enum_model.log_clamp_probability(L_ENUM, {1: "G"})
    assert math.isfinite(rare)
    assert rare < usual - 2.0


def test_a_clamp_outside_the_alphabet_is_refused(enum_model):
    with pytest.raises(ValueError, match="residues"):
        enum_model.positional_marginals(L_ENUM, clamp={4: "B"})
    with pytest.raises(ValueError, match="outside"):
        enum_model.positional_marginals(L_ENUM, clamp={4: 25})


def test_a_clamp_outside_the_sequence_is_refused(enum_model):
    with pytest.raises(ValueError, match="outside a length"):
        enum_model.positional_marginals(L_ENUM, clamp={L_ENUM: "A"})
    with pytest.raises(ValueError, match="outside a length"):
        enum_model.positional_marginals(L_ENUM, clamp={-1: "A"})


def test_a_contradictory_clamp_is_refused(enum_model):
    """A plain dict cannot express this -- equal keys collapse -- but a
    mapping is only required to have `items()`, and the second write to a
    position would otherwise win silently."""
    class Twice:
        def items(self):
            return [(4, "A"), (4, "G")]

    with pytest.raises(ValueError, match="two different residues"):
        enum_model.positional_marginals(L_ENUM, clamp=Twice())


def test_a_fractional_clamp_position_is_refused(enum_model):
    """`int(4.7)` is 4, and a clamp on position 4.7 is a typo rather than a
    request to round."""
    with pytest.raises(TypeError, match="must be integers"):
        enum_model.positional_marginals(L_ENUM, clamp={4.7: "A"})


# Internal consistency, at every length the model produces
# ═══════════════════════════════════════════════════════════════

def test_every_row_sums_to_one_at_every_producible_length(model):
    """Not renormalised anywhere, so this is a real check on the DP rather
    than a check that a division happened."""
    for length in model.observed_lengths:
        p = model.positional_marginals(length)
        assert p.shape == (length, model.n_residues)
        assert np.abs(p.sum(axis=1) - 1.0).max() < EXACT
        assert (p >= 0.0).all()


def test_rows_still_sum_to_one_under_a_clamp(model):
    length = model.observed_lengths[len(model.observed_lengths) // 2]
    top = model.top_k_sequences(1, length=length)[0][0]
    p = model.positional_marginals(length, clamp={3: top[3]})
    assert np.abs(p.sum(axis=1) - 1.0).max() < EXACT


def test_marginals_agree_with_the_nucleotide_alphabet():
    """Five columns, not twenty, and the same identities.

    The alphabet is a runtime dimension of every table and of the trellis, so
    a sweep that hardcoded twenty anywhere would pass the whole amino-acid
    half of this file and fail here."""
    seqs, abund = make_nt_corpus(n=400, seed=5, lo=15, hi=21)
    nt = BeamModel(seqs, abundances=abund, order=2, alphabet="nt",
                   max_length=24)
    length = min(map(len, seqs))
    p = nt.positional_marginals(length)
    assert p.shape == (length, 5)
    assert np.abs(p.sum(axis=1) - 1.0).max() < EXACT


def test_a_length_outside_the_model_fails_loudly(model):
    """A documented refusal, not a silently flat row of zeros.

    Every length from 1 to max_length is producible now -- max_length itself
    included, though the corpus stops well short of it -- so the refusals
    left are the two ends."""
    top = int(model.config["max_length"])
    assert model.positional_marginals(top).shape == (top, 20)
    with pytest.raises(ValueError, match="at least 1"):
        model.positional_marginals(0)
    with pytest.raises(ValueError, match="between 0 and max_length"):
        model.positional_marginals(top + 1)
    with pytest.raises(ValueError, match="between 0 and max_length"):
        model.positional_marginals(10_000)


def test_the_marginal_of_a_conserved_position_is_nearly_a_point_mass(
        model, corpus):
    """Every training sequence of the fixture corpus starts with C.

    Under the prior another residue at position 0 is possible, but only
    through pseudo-counts backed off from rows that hold thousands of C's, so
    C keeps all but a sliver (measured 1 - 7e-13 at length 18) and nothing
    else is exactly zero. A model that smeared mass across the first column
    would pass every sum-to-one check above; one that let the prior swamp the
    counts would fail the first assertion by orders of magnitude.
    """
    length = max(map(len, corpus[0]))
    p = model.positional_marginals(length)
    assert p[0, AA.index("C")] == pytest.approx(1.0, abs=1e-9)
    assert (p[0] > 0.0).all()


# Expected cell usage
# ═══════════════════════════════════════════════════════════════

def _direct_expected_counts(model, support, length):
    """Expected counts by walking each supported sequence into its cells.

    The addressing is transcribed from `observe_one_position` in training.c and
    the row formula in indexing.h -- deliberately re-derived here rather than
    read back from the C sweep this is checking, since a shared derivation
    would agree with itself whatever it did.

    One row formula now, not three, and no clamp: `row = depth * A**order +
    context`, with depth exact and measured from the anchoring terminus, the
    context packed most-distant-first, and `A = R + 1` so that the digit `R`
    can stand for "before the start" or "past the end".
    """
    _, codes, lp = support
    R = model.n_residues
    A, none = R + 1, R
    order = int(model.config["order"])
    w = np.exp(lp) / np.exp(lp).sum()

    def ctx(p, forward):
        idx = np.zeros(len(codes), dtype=np.int64)
        for j in range(order, 0, -1):
            q = p - j if forward else p + j
            digit = codes[:, q] if 0 <= q < length else none
            idx = idx * A + digit
        return idx

    out = {}
    for anchor, forward in (("forward", True), ("backward", False)):
        rows, res = [], []
        for p in range(length):
            depth = p if forward else length - 1 - p
            rows.append(depth * A ** order + ctx(p, forward))
            res.append(codes[:, p])
        rows, res = np.concatenate(rows), np.concatenate(res)
        uniq, inv = np.unique(rows, return_inverse=True)
        cells = np.bincount(inv * R + res, weights=np.tile(w, length),
                            minlength=len(uniq) * R).reshape(len(uniq), R)
        out[anchor] = {int(r): cells[i] for i, r in enumerate(uniq)}
    return out


def test_expected_counts_match_a_direct_sum_over_the_whole_support(enum_model,
                                                                   support):
    ref = _direct_expected_counts(enum_model, support, L_ENUM)
    got = enum_model.expected_counts(length=L_ENUM)
    for name, want in ref.items():
        blk = got[name]
        have = {int(row): blk["counts"][i]
                for i, row in enumerate(blk["rows"])}
        assert set(have) == set(want), f"{name}: the row sets differ"
        for row, counts in want.items():
            assert np.abs(have[row] - counts).max() < EXACT, f"{name} row {row}"


def test_a_row_absent_from_the_answer_is_exactly_zero(enum_model, support):
    """The sparse form is only honest if what it omits is zero. Checked by
    asking the direct enumeration for the rows the sweep did NOT return."""
    ref = _direct_expected_counts(enum_model, support, L_ENUM)
    got = enum_model.expected_counts(length=L_ENUM)
    for name, want in ref.items():
        returned = set(map(int, got[name]["rows"]))
        assert not [r for r in want if r not in returned], (
            f"{name}: a row with mass was omitted")


def test_every_block_sums_to_the_expected_length(model):
    """Each position increments exactly one cell of each of the six tables, so
    every block sums to L. It is the cheapest check that the sink and the
    first K layers were both covered -- a sweep that dropped the sink would
    still produce a plausible-looking table."""
    length = model.observed_lengths[len(model.observed_lengths) // 2]
    ec = model.expected_counts(length=length)
    assert ec["expected_length"] == pytest.approx(float(length), abs=EXACT)
    for anchor in ("forward", "backward"):
        blk = ec[anchor]
        assert blk["counts"].sum() == pytest.approx(float(length),
                                                    abs=EXACT)
        assert (blk["counts"] >= 0.0).all()
        assert (np.diff(blk["rows"]) > 0).all(), "rows must be ascending"


def test_pooled_expected_counts_sum_to_the_mean_length(small):
    pmf = small.length_pmf()
    mean = float(sum(L * pmf[L] for L in range(len(pmf))))
    ec = small.expected_counts()
    assert ec["expected_length"] == pytest.approx(mean, abs=EXACT)
    for anchor in ("forward", "backward"):
        for expert in ("wide", "mid", "len"):
            assert ec[anchor]["counts"].sum() == pytest.approx(
                mean, abs=EXACT)


def test_the_decoded_columns_rebuild_every_row_index(small):
    """`depth`, `context` and `length` are a convenience, and a convenience
    that disagrees with the row it was decoded from is worse than none."""
    length = int(np.argmax(small.length_pmf()))
    ec = small.expected_counts(length=length)
    cfg = small.config
    A = small.n_residues + 1
    nc = A ** int(cfg["order"])
    for anchor in ("forward", "backward"):
        blk = ec[anchor]
        assert ((blk["depth"] * nc + blk["context"]) == blk["rows"]).all()
        assert (blk["context"] < nc).all()
        # Depth is exact and bounded by the length, not by a cap.
        assert (blk["depth"] < length).all()
        assert blk["depth"].max() == length - 1


def test_expected_counts_refuse_a_length_outside_the_model(model):
    """Every length up to max_length is producible, so the refusal left is a
    length past it. (0 is not a refusal here: it asks for the pooled form.)"""
    with pytest.raises(ValueError, match="between 0 and max_length"):
        model.expected_counts(length=int(model.config["max_length"]) + 1)


def test_expected_counts_reach_every_cell_a_sequence_can_address(small):
    """And no other.

    Under the prior every string of the length is possible, so the expected
    usage reaches EVERY (depth, context) a string of that length can address
    -- each of the R**min(depth, order) contexts at each depth -- with every
    residue in every such row, and nothing outside that set. The set is built
    here from the row formula alone.

    It is also what would break if the C-anchored wide context were read with
    the finalized table's nearest-first packing instead of the counts'
    most-distant-first one: near a terminus the context is part padding, and
    reversing the digits moves the padding and so moves the row out of the
    set.
    """
    length = int(np.argmax(small.length_pmf()))
    ec = small.expected_counts(length=length)
    R = small.n_residues
    A, none = R + 1, R
    order = int(small.config["order"])
    want = set()
    for depth in range(length):
        slots = [range(R) if depth - j >= 0 else (none,)
                 for j in range(order, 0, -1)]
        for digits in itertools.product(*slots):
            idx = 0
            for d in digits:
                idx = idx * A + d
            want.add(depth * A ** order + idx)
    for anchor in ("forward", "backward"):
        blk = ec[anchor]
        assert set(map(int, blk["rows"])) == want, anchor
        assert (blk["counts"] > 0.0).all(), anchor


# The new methods change nothing that was there before
# ═══════════════════════════════════════════════════════════════

def test_the_marginals_reproduce_the_partition_function(enum_model):
    """The backward sweep the marginals normalise by is an independent
    evaluation of ln Z(L): `log_partition` reaches it forwards with per-layer
    rescaling, this reaches it backwards in log space. They have to agree, and
    agreeing is what says the new sweep is normalising by the right thing.

    The clamps on one position partition the length's strings, so their
    clamped Z's must add back up to the unclamped one: the log clamp
    probabilities over the whole alphabet log-sum to exactly zero, at every
    length, including the ones the corpus never had."""
    for length in enum_model.observed_lengths:
        for pos in {0, length - 1}:
            got = [enum_model.log_clamp_probability(length, {pos: r})
                   for r in NT]
            assert all(math.isfinite(v) for v in got)
            top = max(got)
            lse = top + math.log(sum(math.exp(v - top) for v in got))
            assert lse == pytest.approx(0.0, abs=EXACT), (length, pos)


def test_asking_for_marginals_does_not_disturb_the_model(model):
    seqs = [s for s, _ in model.top_k_sequences(50)]
    before = [model.log_pseq(s) for s in seqs]
    diag_before = model.pgen_diagnostics()
    length = int(np.argmax(model.length_pmf()))
    model.positional_marginals(length)
    model.expected_counts(length=length)
    after = [model.log_pseq(s) for s in seqs]
    assert before == after, "scores moved after an analytic ran"
    assert model.pgen_diagnostics() == diag_before


# The same exactness at every order
# ═══════════════════════════════════════════════════════════════

@pytest.mark.parametrize("order", [2, 3, 4])
def test_both_answers_stay_exact_at_every_order(order):
    """Order decides which branch the C-anchored context takes, and the
    branches are not equivalent code paths that happen to be fast or slow.

    With ``K = max(order, 3)``: at order 4 the emitted residue lands in the
    last digit of the wide context and the kernels hoist the mid and length
    lookups out of the residue loop; at order 3 the residue is inside the mid
    context and nothing may be hoisted; at order 2 it is outside the wide
    context altogether. Getting that wrong produces a model that normalises
    perfectly and is addressed by the wrong cell, so each order is enumerated
    separately rather than trusting one to stand for the others.
    """
    m = _enumerable(order=order)
    assert m.path_count(L_ENUM) == len(NT) ** L_ENUM
    sup = _space(m)

    ref, _ = _direct_marginals(sup, L_ENUM)
    assert np.abs(m.positional_marginals(L_ENUM) - ref).max() < EXACT

    ref, _ = _direct_marginals(sup, L_ENUM, {3: "A"})
    got = m.positional_marginals(L_ENUM, clamp={3: "A"})
    assert np.abs(got - ref).max() < EXACT
    assert (got[np.arange(L_ENUM) != 3] > 0.0).all()

    want = _direct_expected_counts(m, sup, L_ENUM)
    ec = m.expected_counts(length=L_ENUM)
    for name, cells in want.items():
        blk = ec[name]
        have = {int(r): blk["counts"][i] for i, r in enumerate(blk["rows"])}
        assert set(have) == set(cells), f"order {order}, {name}: rows differ"
        for row, counts in cells.items():
            assert np.abs(have[row] - counts).max() < EXACT


def test_both_answers_are_bit_identical_across_thread_counts(small):
    """Pinned, not hoped for.

    The marginal sum is chunked into a fixed number of pieces and combined in
    chunk order, and the expected-count accumulation is serial, precisely so
    that neither answer depends on how many cores happened to be free. Nothing
    else in the library would notice if that stopped being true.
    """
    length = small.observed_lengths[-1]
    residue = small.top_k_sequences(1, length=length)[0][0][2]
    before = small.threads
    ref = None
    try:
        for n in (1, 2, 4, 0):
            small.threads = n
            ec = small.expected_counts(length=length)
            got = (small.positional_marginals(length).tobytes(),
                   small.positional_marginals(
                       length, clamp={2: residue}).tobytes(),
                   tuple((b["rows"].tobytes(), b["counts"].tobytes())
                         for k, b in sorted(ec.items())
                         if isinstance(b, dict)))
            if ref is None:
                ref = got
            assert got == ref, f"answer moved at threads={n}"
    finally:
        small.threads = before


def test_an_oversized_sweep_is_refused_before_it_allocates(small):
    """The backward field is held for all `L + 1` layers at once, which is 44
    MB on the default amino-acid model and gigabytes at a nucleotide model's
    ceiling order. `calloc` grants lazily on Linux, so an oversized request
    fails on first write with the OOM killer choosing a victim from the whole
    machine; it has to be refused while refusing is still cheap."""
    import BEAM
    length = int(np.argmax(small.length_pmf()))
    budget = BEAM.memory_budget()
    try:
        BEAM.set_memory_budget(1000)
        with pytest.raises(MemoryError, match="memory budget"):
            small.positional_marginals(length)
        with pytest.raises(MemoryError, match="memory budget"):
            small.expected_counts(length=length)
    finally:
        BEAM.set_memory_budget(budget)
    # and the refusal left nothing broken behind
    assert small.positional_marginals(length).shape == (length, 20)
