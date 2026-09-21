"""File-backed construction and scoring.
"""
from __future__ import annotations

import importlib.util
import math
import random

import numpy as np
import pytest

from BEAM import BeamModel

from _helpers import (AA, J_MOTIFS, make_corpus, ANALYSIS_BINS)

# Streaming and file-backed scoring
# ════════════════════════════════════════════════════════════════════════


def test_log_pseq_chunks_reassembles_the_batch(model, corpus):
    seqs, _ = corpus
    probe = seqs[:500]
    chunked = np.concatenate(list(model.log_pseq_chunks(probe, chunk_size=61)))
    assert np.array_equal(chunked, model.log_pseq_batch(probe))


def test_log_pseq_file_matches_the_batch(tmp_path, model, corpus):
    seqs, abund = corpus
    path = tmp_path / "score.tsv"
    path.write_text("".join(f"{s}\t{a}\n" for s, a in zip(seqs, abund)))
    got = model.log_pseq_file(path, chunk_size=97)
    assert np.array_equal(got, model.log_pseq_batch(seqs))


def test_log_pseq_file_fills_a_preallocated_array(tmp_path, model, corpus):
    """The pattern a corpus-scale scan uses: a memmap the size of the corpus,
    filled in place, never a Python list of 900 million floats."""
    seqs, _ = corpus
    path = tmp_path / "plain.txt"
    path.write_text("".join(f"{s}\n" for s in seqs))
    out = np.zeros(len(seqs), dtype=np.float32)
    got = model.log_pseq_file(path, chunk_size=113, out=out)
    assert got is out
    assert np.allclose(out, model.log_pseq_batch(seqs), rtol=1e-6)


def test_log_pseq_file_rejects_a_wrong_sized_out(tmp_path, model, corpus):
    seqs, _ = corpus
    path = tmp_path / "plain.txt"
    path.write_text("".join(f"{s}\n" for s in seqs[:100]))
    with pytest.raises(ValueError, match="not the right length"):
        model.log_pseq_file(path, out=np.zeros(50))
    with pytest.raises(ValueError, match="not the right length"):
        model.log_pseq_file(path, out=np.zeros(500))


def test_log_pseq_parquet_preserves_row_order(tmp_path, model, corpus):
    """Two positionally aligned columns are joined by ROW ORDER and nothing
    else, so a scorer that reordered rows would pair every sequence with some
    other sequence's label."""
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    seqs, _ = corpus
    path = tmp_path / "s.parquet"
    pq.write_table(pa.table({"cdr3_aa": seqs}), path)
    got = model.log_pseq_parquet(path, batch_size=173)
    assert np.array_equal(got, model.log_pseq_batch(seqs))


def test_pseq_accepts_a_list_and_a_log_flag(model, corpus):
    seqs, _ = corpus
    probe = seqs[:20]
    assert model.pseq(probe, log=True) == pytest.approx(
        model.log_pseq_batch(probe))
    assert model.pseq(probe) == pytest.approx(np.exp(model.log_pseq_batch(probe)))
    assert model.pseq(probe[0]) == pytest.approx(math.exp(
        model.log_pseq(probe[0])))
    assert model.pseq(probe[0], log=True) == pytest.approx(
        model.log_pseq(probe[0]))


# ════════════════════════════════════════════════════════════════════════
# pseq_analysis: spectra, atoms and discovery
# ════════════════════════════════════════════════════════════════════════

#: Grid used by the analysis tests. Small enough to keep a transport in
#: seconds on the `small` fixture, fine enough that the grid error is well
#: below every tolerance asserted against it.


@pytest.fixture(scope="module")
def tiny():
    """A model whose ENTIRE support can be enumerated: 20**3 + 20**4 = 168,000
    sequences. Small enough that every spectrum quantity has an exact
    brute-force counterpart to check against.

    Open-world, and it has to be: the enumeration is only the whole support
    while backoff keeps every cell positive. Under the closed-world default
    the support is whatever 500 short sequences happen to spell, which is not
    a set an exhaustive product enumerates."""
    rng = random.Random(3)
    seqs = ["".join(rng.choice(AA) for _ in range(rng.choice([3, 4])))
            for _ in range(500)]
    abund = [rng.randint(1, 9) for _ in seqs]
    return BeamModel(seqs, abundances=abund, order=2,
                     max_length=4, strict=False)


@pytest.fixture(scope="module")
def tiny_exact(tiny):
    """Every sequence the tiny model can produce, with its exact probability."""
    import itertools
    every = ["".join(t) for L in (3, 4)
             for t in itertools.product(AA, repeat=L)]
    p = np.exp(tiny.log_pseq_batch(every).astype(np.longdouble))
    return every, p


def test_histogram_pair_shares_one_grid_and_hits_its_totals(small):
    a = small.pseq_analysis()
    pair = a.histogram_pair(ANALYSIS_BINS)
    c, g = pair["counting"], pair["generated"]
    assert np.array_equal(c.surprisal, g.surprisal)
    assert c.grid_spacing == g.grid_spacing
    assert c.total_mass == pytest.approx(small.path_count(), rel=1e-9)
    assert g.total_mass == pytest.approx(1.0, rel=1e-12)


def test_histogram_mean_is_the_entropy(small):
    """Linear transport conserves the FIRST moment exactly, so the mean read
    off the grid is not an approximation even though the shape is."""
    a = small.pseq_analysis()
    h = a.histogram(ANALYSIS_BINS, measure="generated")
    assert h.mean == pytest.approx(small.entropy(), rel=1e-9)


def test_histogram_is_cached_not_rebuilt(small):
    a = small.pseq_analysis()
    assert a.histogram(ANALYSIS_BINS) is a.histogram(ANALYSIS_BINS)
    assert a.histogram(ANALYSIS_BINS) is not a.histogram(ANALYSIS_BINS + 1)


def test_histograms_by_length_decompose_the_pooled_spectrum(small):
    a = small.pseq_analysis()
    per_length = a.histograms_by_length(ANALYSIS_BINS, measure="generated")
    marg = a.length_marginals()
    assert set(per_length) == set(marg)
    for L, h in per_length.items():
        assert h.total_mass == pytest.approx(marg[L]["generated"], rel=1e-9)
    total = sum(h.total_mass for h in per_length.values())
    assert total == pytest.approx(1.0, rel=1e-12)
    # And they all sit on the pooled grid, which is what makes them stackable.
    pooled = a.histogram(ANALYSIS_BINS, measure="generated")
    for h in per_length.values():
        assert np.array_equal(h.surprisal, pooled.surprisal)


def test_histograms_by_length_respects_max_length(small):
    a = small.pseq_analysis()
    got = a.histograms_by_length(ANALYSIS_BINS, max_length=7)
    assert got and max(got) <= 7


def test_atom_mass_is_exactly_one_where_exp_surprisal_is_not(open_small):
    """The atom probability is `generated / counting`, not `exp(-surprisal)`.

    Both spectra are transported on the same grid with the same linear
    splitting, so their ratio is the mass-weighted mean probability of the
    sequences in the bin and `sum(multiplicity * probability)` is the
    generated mass EXACTLY. Reading exp(-surprisal) off the bin's left edge
    instead biases every atom by up to exp(spacing), which this test pins
    down: at a coarse grid the naive route is out by tens of percent, and
    every discovery curve built on it inherits that.
    """
    a = open_small.pseq_analysis()
    for bins in (128, ANALYSIS_BINS):
        atoms = a.atoms(bins)
        assert atoms.mass == pytest.approx(1.0, rel=1e-12)
        assert atoms.support_size == pytest.approx(open_small.path_count(), rel=1e-9)

        h = a.histogram(bins, measure="counting")
        keep = h.weights > 0
        naive = float(np.sum(
            h.weights[keep].astype(np.longdouble)
            * np.exp(-h.surprisal[keep].astype(np.longdouble))))
        assert naive > 1.01, (
            "the naive exp(-surprisal) route is supposed to be biased here; "
            "if it is not, this test is no longer proving anything")


def test_discovery_curve_matches_exhaustive_enumeration(tiny, tiny_exact):
    """R(n) and U(n) from the grid spectrum against the same sums taken over
    every one of the 168,000 sequences the model can produce.

    This is the check that the atoms really describe the distribution rather
    than merely being self-consistent: the right-hand side never touches a
    spectrum.
    """
    _, p = tiny_exact
    atoms = tiny.pseq_analysis().atoms(4096)
    assert atoms.support_size == pytest.approx(20 ** 3 + 20 ** 4, rel=1e-9)

    for n, tol in ((1, 1e-9), (10, 1e-5), (1_000, 1e-4),
                   (100_000, 1e-3), (10 ** 8, 1e-3)):
        want_r = float(np.sum(-np.expm1(np.longdouble(n) * np.log1p(-p))))
        want_u = float(np.sum(p * np.exp(np.longdouble(n - 1)
                                         * np.log1p(-p))))
        assert atoms.expected_richness(n) == pytest.approx(want_r, rel=tol)
        assert atoms.novelty(n) == pytest.approx(want_u, rel=tol)


def test_discovery_curve_refines_with_the_grid(tiny, tiny_exact):
    """The grid is the only approximation, so the error must fall as it is
    refined. A bug that biased the atoms would leave a floor instead."""
    _, p = tiny_exact
    n = 100_000
    want = float(np.sum(-np.expm1(np.longdouble(n) * np.log1p(-p))))
    a = tiny.pseq_analysis()
    errors = [abs(a.atoms(b).expected_richness(n) - want) / want
              for b in (512, 2048, 8192)]
    assert errors[0] > errors[1] > errors[2]
    assert errors[-1] < 1e-5


def test_novelty_is_the_richness_increment(small):
    """U(n) == R(n) - R(n-1) identically. Two different sums over the same
    atoms, so a sign or off-by-one in either shows up here."""
    atoms = small.pseq_analysis().atoms(ANALYSIS_BINS)
    for n in (2, 7, 50, 1000):
        assert atoms.novelty(n) == pytest.approx(
            atoms.expected_richness(n) - atoms.expected_richness(n - 1),
            rel=1e-9)


def test_expected_richness_at_one_draw_is_the_mass(small):
    atoms = small.pseq_analysis().atoms(ANALYSIS_BINS)
    assert atoms.expected_richness(1) == pytest.approx(1.0, rel=1e-9)
    assert atoms.novelty(1) == pytest.approx(1.0, rel=1e-9)
    assert atoms.expected_richness(0) == pytest.approx(0.0, abs=1e-12)


def test_expected_richness_is_monotone_and_bounded(small):
    atoms = small.pseq_analysis().atoms(ANALYSIS_BINS)
    d0 = small.path_count()
    ns = np.logspace(0, 60, 40)
    r = [atoms.expected_richness(n) for n in ns]
    assert all(b >= a for a, b in zip(r, r[1:]))
    assert all(v <= d0 * (1 + 1e-9) for v in r)
    assert atoms.expected_richness(1e300) == pytest.approx(d0, rel=1e-6)


def test_expected_frequency_partitions_the_support(small):
    """Every sequence is drawn 0..n times, so the frequencies sum to D0."""
    atoms = small.pseq_analysis().atoms(ANALYSIS_BINS)
    n = 6
    total = sum(atoms.expected_frequency(n, r) for r in range(n + 1))
    assert total == pytest.approx(small.path_count(), rel=1e-6)
    # f0 is what R(n) says is still unseen.
    assert atoms.expected_frequency(n, 0) == pytest.approx(
        small.path_count() - atoms.expected_richness(n), rel=1e-6)


def test_coverage_matches_exhaustive_enumeration(tiny, tiny_exact):
    _, p = tiny_exact
    atoms = tiny.pseq_analysis().atoms(4096)
    for n in (10, 1_000, 100_000):
        want = float(np.sum(p * -np.expm1(np.longdouble(n) * np.log1p(-p))))
        assert atoms.coverage(n) == pytest.approx(want, rel=1e-3)


def test_discovery_curve_evaluates_many_depths_at_once(small):
    atoms = small.pseq_analysis().atoms(ANALYSIS_BINS)
    curve = atoms.discovery_curve([1, 10, 100])
    assert curve["expected_richness"][0] == pytest.approx(
        atoms.expected_richness(1), rel=1e-12)
    assert curve["novelty"][2] == pytest.approx(atoms.novelty(100), rel=1e-12)
    assert list(curve) == ["n", "expected_richness", "novelty", "coverage"]


def test_atoms_at_one_length_carry_that_length_s_mass(small):
    a = small.pseq_analysis()
    marg = a.length_marginals()
    L = max(marg, key=lambda k: marg[k]["generated"])
    atoms = a.atoms(ANALYSIS_BINS, length=L)
    assert atoms.mass == pytest.approx(marg[L]["generated"], rel=1e-9)
    assert atoms.support_size == pytest.approx(small.path_count(L), rel=1e-9)


# ════════════════════════════════════════════════════════════════════════
