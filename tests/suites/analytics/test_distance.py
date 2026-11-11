"""Local repertoire distance against independent complete-sequence sums.

The reference below never reads transition rows or the distance kernel. It
enumerates all strings of tiny nucleotide models and aggregates their scored
probabilities into consecutive words.
"""
import itertools
import math
from dataclasses import replace

import numpy as np
import pytest

import BEAM
from BEAM import BeamModel, BeamQuery
from BEAM.analytics.distance import (
    DistanceComparison,
    LengthContribution,
    PatternContribution,
    WindowContribution,
    compare,
    group_overlaps,
)
from BEAM.model.alphabets import symbols


def tiny_model(reverse=False, *, order=2, weights=(0.7, 0.4), max_length=4):
    sequences = ["A", "C", "AC", "TN", "ACGT", "CCGT", "AGTA", "TTAC", "GCNA", "ANNC"]
    counts = [3, 2, 7, 5, 29, 17, 13, 11, 7, 5]
    if reverse:
        counts = counts[::-1]
    return BeamModel(sequences, abundances=counts, alphabet="nt", order=order,
                     max_length=max_length, alpha=2.0, weights=weights)


@pytest.fixture(scope="module")
def pair():
    return tiny_model(), tiny_model(True)


def complete_law(model, length):
    sequences = ["".join(x) for x in itertools.product(symbols(model.alphabet), repeat=length)]
    conditional = np.exp(model.log_pseq_batch(sequences) - model.log_length_prior(length))
    assert conditional.sum() == pytest.approx(1.0, abs=3e-12)
    return sequences, conditional


def reference_distance(a, b, width):
    pa, pb = a.length_pmf(), b.length_pmf()
    lengths, windows, words = {}, {}, {}
    for length in range(1, max(len(pa), len(pb))):
        p = float(pa[length]) if length < len(pa) else 0.0
        q = float(pb[length]) if length < len(pb) else 0.0
        if p == q == 0:
            continue
        lengths[length] = 0.5 * (math.sqrt(p) - math.sqrt(q)) ** 2
        block = min(width, length)
        n_windows = length - block + 1
        if p == 0 or q == 0:
            for position in range(n_windows):
                windows[length, position] = 0.0
            continue
        sequences, pp = complete_law(a, length)
        _, qq = complete_law(b, length)
        coefficient = math.sqrt(p * q) / (2 * n_windows)
        for position in range(n_windows):
            marginal = {}
            for sequence, pr, qr in zip(sequences, pp, qq):
                word = sequence[position:position + block]
                v = marginal.setdefault(word, [0., 0.])
                v[0] += pr
                v[1] += qr
            terms = []
            for word, (pr, qr) in marginal.items():
                contribution = coefficient * (math.sqrt(pr) - math.sqrt(qr)) ** 2
                words[length, position, word] = (pr, qr, contribution)
                terms.append(contribution)
            windows[length, position] = math.fsum(terms)
    return lengths, windows, words


@pytest.mark.parametrize("order,width,weights", [
    (2, None, (0.7, 0.4)),
    (2, 1, (0.7, 0.4)),
    (2, 2, (1.0, 0.0)),
    (4, None, (0.0, 1.0)),
])
def test_complete_sequence_oracle(order, width, weights):
    a = tiny_model(order=order, weights=weights)
    b = tiny_model(True, order=order, weights=weights)
    result = compare(a, b, width=width, top_k=10_000)
    chosen_width = order + 1 if width is None else width
    lengths, windows, words = reference_distance(a, b, chosen_width)
    assert result.width == chosen_width
    assert result.length_squared_distance == pytest.approx(math.fsum(lengths.values()), abs=2e-13)
    assert result.pattern_squared_distance == pytest.approx(math.fsum(windows.values()), abs=2e-13)
    assert result.squared_distance == pytest.approx(
        math.fsum(lengths.values()) + math.fsum(windows.values()), abs=2e-13)
    assert result.distance ** 2 == pytest.approx(result.squared_distance, abs=2e-15)
    assert {v.length for v in result.lengths} == set(lengths)
    assert {(v.length, v.start) for v in result.windows} == set(windows)
    for row in result.lengths:
        assert row.contribution == pytest.approx(lengths[row.length], abs=2e-13)
        expected = math.fsum(v for (length, _), v in windows.items() if length == row.length)
        assert row.pattern_contribution == pytest.approx(expected, abs=2e-13)
    for row in result.windows:
        assert row.width == min(chosen_width, row.length)
        assert row.contribution == pytest.approx(windows[row.length, row.start], abs=2e-13)
    for row in result.patterns:
        p, q, contribution = words[row.length, row.start, row.word]
        assert row.conditional_p == pytest.approx(p, rel=2e-11, abs=2e-13)
        assert row.conditional_q == pytest.approx(q, rel=2e-11, abs=2e-13)
        assert row.contribution == pytest.approx(contribution, rel=2e-10, abs=2e-15)
        assert row.delta == pytest.approx(p - q, abs=2e-13)
        assert row.direction == ("p" if row.conditional_p > row.conditional_q else "q")
    assert result.remainder == pytest.approx(0.0, abs=3e-13)
    assert result.coverage == pytest.approx(1.0, abs=3e-13)


def test_identity_swap_thread_invariance_and_reconstruction(pair):
    a, b = pair
    initial_threads = BEAM.threads()
    try:
        BEAM.set_threads(1)
        result = compare(a, b, top_k=23)
        swapped = compare(b, a, top_k=23)
        same = compare(a, a, top_k=23)
        BEAM.set_threads(2)
        threaded = compare(a, b, top_k=23)
    finally:
        BEAM.set_threads(initial_threads)
    assert result.squared_distance == swapped.squared_distance == threaded.squared_distance
    assert result.patterns == threaded.patterns
    assert same.squared_distance == 0.0
    assert same.patterns == ()
    assert same.coverage is None
    assert same.total_coverage is None
    assert same.remainder == 0.0
    for left, right in zip(result.patterns, swapped.patterns):
        assert (left.length, left.start, left.word) == (right.length, right.start, right.word)
        assert left.conditional_p == right.conditional_q
        assert left.conditional_q == right.conditional_p
        assert left.contribution == right.contribution
        assert left.direction != right.direction
    assert result.squared_distance == pytest.approx(
        result.length_squared_distance + math.fsum(p.contribution for p in result.patterns)
        + result.remainder, abs=2e-14)


@pytest.mark.parametrize("top_k", [0, 1, 7, 10_000])
def test_global_top_k_leaves_full_distance_unchanged(pair, top_k):
    a, b = pair
    full = compare(a, b, top_k=10_000)
    result = compare(a, b, top_k=top_k)
    assert result.squared_distance == full.squared_distance
    assert result.length_squared_distance == full.length_squared_distance
    assert result.pattern_squared_distance == full.pattern_squared_distance
    assert result.patterns == full.patterns[:top_k]
    reported = math.fsum(p.contribution for p in result.patterns)
    assert result.reported_pattern_squared_distance == pytest.approx(reported, abs=2e-15)
    assert result.remainder == pytest.approx(full.pattern_squared_distance - reported, abs=2e-14)
    assert result.coverage == pytest.approx(reported / full.pattern_squared_distance, abs=2e-14)
    assert result.total_coverage == pytest.approx(
        (result.length_squared_distance + reported) / full.squared_distance, abs=2e-14)
    assert all(p.contribution > 0 for p in result.patterns)


def test_deterministic_tie_order():
    # Complementary weights give exactly equal contributions for A and C.
    a = BeamModel(["A", "C"], abundances=[9, 1], alphabet="nt", order=2,
                  max_length=2, alpha_length=1.)
    b = BeamModel(["A", "C"], abundances=[1, 9], alphabet="nt", order=2,
                  max_length=2, alpha_length=1.)
    full = compare(a, b, top_k=20)
    tied = [p for p in full.patterns if p.word in ("A", "C")]
    assert len(tied) == 2
    assert tied[0].contribution == tied[1].contribution
    assert [p.word for p in tied] == ["A", "C"]
    assert compare(a, b, top_k=1).patterns == full.patterns[:1]


def test_distinct_live_length_support_and_different_maximum_lengths():
    a = BeamModel(["A", "C", "AA"], abundances=[4, 2, 3], alphabet="nt", order=2,
                  max_length=2, alpha_length=1.)
    b = BeamModel(["A", "C", "ACG"], abundances=[1, 7, 5], alphabet="nt", order=2,
                  max_length=4, alpha_length=1.)
    result = compare(a, b, top_k=100)
    lengths, windows, _ = reference_distance(a, b, 3)
    assert {row.length for row in result.lengths} == {1, 2, 3, 4}
    assert result.squared_distance == pytest.approx(
        math.fsum(lengths.values()) + math.fsum(windows.values()), abs=3e-13)
    assert {p.length for p in result.patterns} == {1, 2}
    assert all(w.contribution == 0 for w in result.windows if w.length in (3, 4))
    rows = {row.length: row for row in result.lengths}
    assert rows[2].q > 0.0  # The positive length prior also supports unseen lengths.
    assert rows[3].p == 0.0
    assert rows[4].p == 0.0


def test_single_position_width_does_not_claim_full_sequence_separation():
    sequences = ["".join(s) for s in itertools.product("AC", repeat=3)]
    counts = [900 if s.count("C") % 2 == 0 else 100 for s in sequences]
    config = dict(alphabet="nt", order=2, max_length=3, weights=(1., 0.),
                  alpha=1e-6, alpha_length=1e-6)
    a = BeamModel(sequences, abundances=counts, **config)
    b = BeamModel(sequences, abundances=[1000 - n for n in counts], **config)
    assert compare(a, b, width=1).squared_distance < 1e-12
    assert compare(a, b).squared_distance > 0.39


@pytest.mark.parametrize("kwargs", [
    {"width": 0}, {"width": -1}, {"width": 4}, {"width": 1.5}, {"width": True},
    {"top_k": -1}, {"top_k": 1.5}, {"top_k": True}, {"top_k": 2**32},
])
def test_invalid_options_refused(pair, kwargs):
    expected = TypeError if any(isinstance(v, (bool, float)) for v in kwargs.values()) else ValueError
    with pytest.raises(expected):
        compare(*pair, **kwargs)


def test_incompatible_models_refused(pair):
    a, _ = pair
    aa = BeamModel(["AC"], order=2, max_length=4)
    different_order = tiny_model(order=3)
    for b in (aa, different_order):
        with pytest.raises(BEAM.ModelMismatch):
            compare(a, b)
    with pytest.raises((ValueError, TypeError)):
        compare(a, object())


def test_memory_budget_refusal(pair):
    initial = BEAM.memory_budget()
    try:
        BEAM.set_memory_budget(1024)
        with pytest.raises(BEAM.BudgetExceeded):
            compare(*pair)
    finally:
        BEAM.set_memory_budget(initial)


def explanation(patterns, *, alphabet="nt", remainder=0.11):
    """Synthetic attribution coordinates for testing only the grouping rules."""
    patterns = tuple(patterns)
    width = len(patterns[0].word)
    reported = math.fsum(p.contribution for p in patterns)
    support = sorted({p.length for p in patterns})
    lengths = tuple(LengthContribution(
        length, 1 / len(support), 1 / len(support), 0.,
        math.fsum(p.contribution for p in patterns if p.length == length)
        + remainder / len(support)) for length in support)
    coordinates = sorted({(p.length, p.start) for p in patterns})
    windows = tuple(WindowContribution(
        length, start, width,
        math.fsum(p.contribution for p in patterns if (p.length, p.start) == (length, start)))
        for length, start in coordinates)
    return DistanceComparison(
        alphabet=alphabet, order=width - 1, width=width, top_k=len(patterns),
        squared_distance=reported + remainder, length_squared_distance=0.,
        pattern_squared_distance=reported + remainder,
        reported_pattern_squared_distance=reported, remainder=remainder,
        scratch_bytes=0, lengths=lengths, windows=windows, patterns=patterns)


def term(length, start, word, contribution, *, reverse=False):
    p, q = (0.05, 0.20) if reverse else (0.20, 0.05)
    return PatternContribution(length, start, word, p, q, contribution)


def test_five_overlapping_windows_form_one_consensus_without_changing_accounting():
    patterns = [term(15, start, word, amount) for start, word, amount in [
        (8, "SNERL", 0.05), (7, "ISNER", 0.04), (9, "NERLF", 0.03),
        (10, "ERLFF", 0.02), (6, "TISNE", 0.01),
    ]]
    result = explanation(patterns, alphabet="aa")
    grouped = group_overlaps(result)
    assert grouped == result.group_overlaps()
    assert grouped.min_overlap == 4
    assert len(grouped.groups) == 1
    group = grouped.groups[0]
    assert (group.length, group.start, group.end, group.consensus) == (15, 6, 15, "TISNERLFF")
    assert group.width == 9
    assert group.direction == "p"
    assert set(group.members) == set(patterns)
    assert group.contribution == math.fsum(p.contribution for p in patterns)
    assert grouped.distance == result.distance
    assert grouped.squared_distance == result.squared_distance
    assert grouped.remainder == result.remainder
    assert grouped.coverage == result.coverage
    assert grouped.total_coverage == result.total_coverage
    assert grouped.reported_pattern_squared_distance == result.reported_pattern_squared_distance
    assert grouped.squared_distance == pytest.approx(
        grouped.length_squared_distance + sum(g.contribution for g in grouped.groups)
        + grouped.remainder, abs=1e-15)


def test_bridge_cannot_merge_contradictions_or_opposite_directions_or_lengths():
    patterns = [
        term(4, 0, "ACG", 0.05),
        term(4, 1, "CGT", 0.04),
        term(4, 0, "TCG", 0.03),  # Matches the bridge, conflicts with the full consensus.
        term(4, 1, "CGT", 0.02, reverse=True),
        term(5, 0, "ACG", 0.01),
    ]
    # Coordinates in a real comparison are unique. Opposite direction is tested
    # at a different word that shares the compatible suffix with the first group.
    patterns[3] = term(4, 1, "CGN", 0.02, reverse=True)
    grouped = group_overlaps(explanation(patterns))
    assert len(grouped.groups) == 4
    assert grouped.groups[0].consensus == "ACGT"
    assert len(grouped.groups[0].members) == 2
    flattened = [p for g in grouped.groups for p in g.members]
    assert len(flattened) == len(set(flattened)) == len(patterns)
    assert set(flattened) == set(patterns)
    for group in grouped.groups:
        assert {p.direction for p in group.members} == {group.direction}
        assert {p.length for p in group.members} == {group.length}
        for member in group.members:
            offset = member.start - group.start
            assert group.consensus[offset:offset + member.width] == member.word
    assert grouped.groups == group_overlaps(
        replace(explanation(patterns), patterns=tuple(reversed(patterns)))).groups


def test_direction_rule_separates_compatible_overlapping_words():
    result = explanation([
        term(4, 0, "ACG", 0.05), term(4, 1, "CGT", 0.04, reverse=True)])
    groups = group_overlaps(result).groups
    assert len(groups) == 2
    assert {g.direction for g in groups} == {"p", "q"}


def test_late_compatible_bridge_merges_previously_separate_groups():
    result = explanation([
        term(5, 0, "ACG", 0.05), term(5, 2, "GTA", 0.04), term(5, 1, "CGT", 0.03)])
    grouped = group_overlaps(result)
    assert len(grouped.groups) == 1
    assert grouped.groups[0].consensus == "ACGTA"
    assert set(grouped.groups[0].members) == set(result.patterns)
    assert grouped.reported_pattern_squared_distance == result.reported_pattern_squared_distance


def test_overlap_threshold_changes_only_grouping():
    result = explanation([term(5, 0, "ACG", 0.05), term(5, 2, "GTA", 0.04)])
    strict = group_overlaps(result)
    relaxed = group_overlaps(result, min_overlap=1)
    assert len(strict.groups) == 2
    assert len(relaxed.groups) == 1
    assert relaxed.groups[0].consensus == "ACGTA"
    assert relaxed.squared_distance == strict.squared_distance
    assert relaxed.reported_pattern_squared_distance == strict.reported_pattern_squared_distance


@pytest.mark.parametrize("minimum,error", [(0, ValueError), (4, ValueError), (1.5, TypeError), (True, TypeError)])
def test_invalid_overlap_threshold_refused(minimum, error):
    result = explanation([term(4, 0, "ACG", 0.05)])
    with pytest.raises(error):
        group_overlaps(result, min_overlap=minimum)


def test_duplicate_coordinates_refused_in_grouping():
    repeated = term(4, 0, "ACG", 0.05)
    with pytest.raises(ValueError, match="once"):
        group_overlaps(explanation([repeated, repeated]))


def test_empty_explanation_stays_empty(pair):
    result = compare(*pair, top_k=0)
    grouped = group_overlaps(result)
    assert grouped.groups == ()
    assert grouped.reported_pattern_squared_distance == 0
    assert grouped.remainder == result.pattern_squared_distance


def test_complete_consensus_probability_is_a_separate_event():
    config = dict(alphabet="nt", order=2, max_length=4, alpha=2.)
    a = BeamModel(["AAAA", "CCCC"], abundances=[9, 1], **config)
    b = BeamModel(["AAAA", "CCCC"], abundances=[1, 9], **config)
    comparison = compare(a, b, top_k=10_000)
    grouped = group_overlaps(comparison)
    group = next(g for g in grouped.groups if g.consensus == "AAAA")
    assert len(group.members) == 2
    original_contribution = group.contribution
    probabilities = group.probabilities(a, b)
    for model, side in ((a, "p"), (b, "q")):
        sequences, law = complete_law(model, group.length)
        expected = math.fsum(p for s, p in zip(sequences, law)
                             if s[group.start:group.end] == group.consensus)
        query = BeamQuery(model)
        condition = query.condition(group.length)
        for offset, residue in enumerate(group.consensus):
            condition = condition.at(group.start + offset, residue)
        conditional = getattr(probabilities, "conditional_" + side)
        assert conditional == pytest.approx(expected, rel=2e-11, abs=2e-13)
        assert conditional == pytest.approx(query.probability(condition), abs=2e-13)
        assert getattr(probabilities, side) == pytest.approx(
            expected * math.exp(model.log_length_prior(group.length)), abs=2e-13)
        summed_window_probabilities = sum(getattr(p, "conditional_" + side) for p in group.members)
        assert not math.isclose(conditional, summed_window_probabilities, rel_tol=1e-3)
    assert group.contribution == original_contribution
    assert group.contribution == math.fsum(p.contribution for p in group.members)


def test_group_probability_unsupported_length_is_not_a_defined_conditional(pair):
    a, _ = pair
    shorter = BeamModel(["AC"], alphabet="nt", order=2, max_length=2)
    group = group_overlaps(explanation([term(4, 0, "ACG", 0.05)])).groups[0]
    probabilities = group.probabilities(a, shorter)
    assert probabilities.conditional_p > 0
    assert probabilities.conditional_q is None
    assert probabilities.q == probabilities.length_q == 0.0


def test_member_enrichment_does_not_imply_complete_consensus_enrichment():
    # P has many AAC and CAA sequences, so each local AA window is more
    # probable in P. Q has more complete AAA sequences, reversing the sign
    # when both window constraints are imposed together.
    sequences = ["AAC", "CAA", "AAA", "CCC"]
    config = dict(alphabet="nt", order=2, max_length=3, weights=(1., 0.),
                  alpha=1e-6, alpha_length=1e-6)
    a = BeamModel(sequences, abundances=[45, 45, 10, 1], **config)
    b = BeamModel(sequences, abundances=[1, 1, 40, 58], **config)
    full = compare(a, b, width=2, top_k=1000)
    members = tuple(p for p in full.patterns if p.length == 3 and p.word == "AA")
    assert len(members) == 2
    assert all(p.direction == "p" for p in members)
    reported = math.fsum(p.contribution for p in members)
    selected = replace(full, patterns=members, top_k=2,
                       reported_pattern_squared_distance=reported,
                       remainder=full.pattern_squared_distance - reported)
    group, = group_overlaps(selected).groups
    assert group.consensus == "AAA"
    assert group.direction == "p"  # Describes the member coordinates only.
    probabilities = group.probabilities(a, b)
    assert probabilities.conditional_p < 0.11
    assert probabilities.conditional_q > 0.39
    assert probabilities.conditional_p < probabilities.conditional_q
    assert group.contribution == reported
