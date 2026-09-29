"""Configuration guards, argument validation, return conventions.
"""
from __future__ import annotations

import importlib.util
import math
import random

import numpy as np
import pytest

from BEAM import BeamModel
from BEAM import _cbeam as _c
from BEAM._model import _config_args_from_dict

from _helpers import AA, J_MOTIFS, make_corpus

# ── Configuration guards ─────────────────────────────────────────

def test_single_anchor_weights_are_accepted(corpus):
    """(1, 0) and (0, 1) must build, and are the point.

    The weights should sum to 1 and are deliberately NOT refused when they do
    not, because scoring each anchor's chain alone is how a partition
    function below 1 is traced to its cause. Every row of either table is a
    full categorical under the prior, so one anchor alone is a proper chain
    and its Z(L) is exactly one: whatever the pair loses is the two chains
    disagreeing, and nothing else. A refusal here would make that
    unmeasurable.
    """
    seqs, abund = corpus
    for w in ((1.0, 0.0), (0.0, 1.0)):
        m = BeamModel(seqs, abundances=abund, order=2, weights=w)
        assert m.config["weights"] == w
        for L in (1, len(seqs[0]), int(m.config["max_length"])):
            assert m.log_partition(L) == pytest.approx(0.0, abs=1e-12), L
    both = BeamModel(seqs, abundances=abund, order=2)
    assert both.log_partition(len(seqs[0])) < -1e-6


def test_rejects_bad_order(corpus):
    seqs, abund = corpus
    with pytest.raises(ValueError):
        BeamModel(seqs, abundances=abund, order=5)


def test_requires_sequences():
    with pytest.raises(ValueError):
        BeamModel([])


def test_skips_unusable_sequences():
    m = BeamModel(["CASSQYF", "CASSXYF", "CASSEQYF"], order=2)
    assert m.n_observed == 2
    assert m.n_skipped == 1


def test_compact_frees_counts(corpus):
    seqs, abund = corpus
    m = BeamModel(seqs, abundances=abund, order=2)
    before = m.memory_bytes
    score = m.log_pseq(seqs[0])
    m.compact()
    assert m.memory_bytes < before
    assert m.log_pseq(seqs[0]) == pytest.approx(score, abs=1e-12)


# ═══════════════════════════════════════════════════════════════
# Argument validation
#
# The C entry points take these as uint8_t and uint32_t and do not populate
# the library's error-message buffer, so an out-of-range value is either
# silently truncated into a different model or surfaces as a bare "invalid
# argument". Both are checked here because both were live.
# ═══════════════════════════════════════════════════════════════

@pytest.mark.parametrize("kwargs,field", [
    (dict(order=260), "order"),          # (uint8_t)260 == 4, the default
    (dict(order=0), "order"),
    (dict(max_length=283), "max_length"),   # (uint8_t)283 == 27
    (dict(max_length=1), "max_length"),
    (dict(max_length=0), "max_length"),
])
def test_out_of_range_configuration_is_refused(kwargs, field):
    """Truncation to uint8 must not silently build a different model."""
    base = dict(order=2, max_length=6)
    base.update(kwargs)
    with pytest.raises(ValueError, match=field):
        BeamModel(["CASF"], **base)


def test_unusable_corpus_says_why():
    """Every sequence rejected is the overwhelmingly common build failure, and
    the C layer reports it as "invalid argument"."""
    with pytest.raises(ValueError, match="survived encoding"):
        BeamModel(["XXXX", "ZZZ", ""], order=2, max_length=6)
    with pytest.raises(ValueError, match="survived encoding"):
        BeamModel(["CASF"], abundances=[0.0], order=2, max_length=6)


def test_negative_counts_are_refused(small):
    """`sample(-1)` reached the C layer as 4,294,967,295 draws, which does not
    raise: it allocates tens of gigabytes and never returns."""
    with pytest.raises(ValueError, match="non-negative"):
        small.sample(-1)
    with pytest.raises(ValueError, match="non-negative"):
        small.top_k_sequences(-1)
    assert small.sample(0) == []
    assert small.top_k_sequences(0) == []


def test_out_of_range_length_is_refused(small):
    """A length outside the model reached the C layer as a huge uint32 and
    came back as a silent -inf or a bare "invalid argument"."""
    hi = int(small.config["max_length"])
    for call in (lambda L: small.log_partition(L),
                 lambda L: small.log_length_prior(L),
                 lambda L: small.path_count(L),
                 lambda L: small.topology(L),
                 lambda L: small.sample(1, length=L),
                 lambda L: small.top_k_sequences(1, length=L),
                 lambda L: small.spectrum(length=L, bins=64)):
        with pytest.raises(ValueError, match="length"):
            call(-1)
        with pytest.raises(ValueError, match="length"):
            call(hi + 1)


def test_topology_refuses_length_zero_and_describes_an_unseen_length(small):
    """Zero means "all lengths" everywhere else, but there is no pooled
    trellis to describe. A length the corpus never had is NOT refused: the
    length prior gives it a share, so it has a trellis like any other."""
    with pytest.raises(ValueError, match="at least 1"):
        small.topology(0)
    unseen = 3                       # the `small` corpus spans 6..9
    assert math.isfinite(small.log_length_prior(unseen))
    t = small.topology(unseen)
    assert t["length"] == unseen and t["n_nodes"] > 0


def test_spectrum_bin_floor_is_explained(small):
    """The nominal floor is 2, but the transport reserves max_length + 9 bins
    of headroom, so the real floor is higher and the C message did not say so.
    An explicit spacing lifts the restriction, and that must keep working."""
    floor = int(small.config["max_length"]) + 10
    with pytest.raises(ValueError, match=f"at least {floor}"):
        small.spectrum(bins=floor - 1)
    with pytest.raises(ValueError, match="at least 2"):
        small.spectrum(bins=1)
    with pytest.raises(ValueError, match="spacing must be positive"):
        small.spectrum(spacing=-1.0)
    assert small.spectrum(bins=4, spacing=50.0)["weights"].size == 4
    with pytest.raises(ValueError, match=f"at least {floor}"):
        small.calibrate(bins=floor - 1)


def test_moment_order_is_range_checked(small):
    with pytest.raises(ValueError, match="order"):
        small.surprisal_moments(order=17)
    with pytest.raises(ValueError, match="order"):
        small.surprisal_cumulants(order=0)


def test_log_base_is_range_checked(small):
    for base in (1.0, 0.0, -2.0):
        with pytest.raises(ValueError, match="base"):
            small.entropy(base=base)
        with pytest.raises(ValueError, match="base"):
            small.dynamic_range(base=base)


# ═══════════════════════════════════════════════════════════════
# explain
# ═══════════════════════════════════════════════════════════════

def test_explain_reports_the_reason_it_refuses(small):
    """Three different refusals share one C error code, so the wrapper has to
    tell them apart or the caller gets "invalid argument" for all three."""
    with pytest.raises(ValueError, match="alphabet does not accept"):
        small.explain("CASSXQYF")
    with pytest.raises(ValueError, match="empty"):
        small.explain("")
    with pytest.raises(ValueError, match="longer than max_length"):
        small.explain("C" * 40)
    with pytest.raises(TypeError):
        small.explain(None)


def test_explain_calls_an_unseen_length_producible(small):
    """A length the corpus never had is priced by the length prior, not ruled
    out by it, so `explain` must present it as an ordinary finite score
    rather than flag it as the explanation of a zero."""
    unseen = 3                       # the `small` corpus spans 6..9
    e = small.explain("C" * unseen)
    assert e["length_is_producible"] is True
    assert np.isfinite(e["forward"]).all() and np.isfinite(e["backward"]).all()
    assert math.isfinite(e["log_pseq"])
    assert e["log_pseq"] == small.log_pseq("C" * unseen)

    live = int(np.argmax(small.length_pmf()))
    good = small.sample(1, length=live, seed=1)[0]
    assert small.explain(good)["length_is_producible"] is True


# ═══════════════════════════════════════════════════════════════
# Return-type conventions
# ═══════════════════════════════════════════════════════════════

def test_scalar_arguments_return_scalars(small):
    """One convention across the module: a scalar in, a float out; anything
    with a shape in, an array of that shape out. `surprisal_pdf` and
    `surprisal_cdf` used to return a one-element array for a scalar while
    `saddlepoint` and `pvalue` returned a float."""
    mu = float(small.surprisal_moments(order=1)[1])
    cal = small.calibrate(bins=64)
    live = int(np.argmax(small.length_pmf()))

    assert isinstance(small.surprisal_pdf(mu), float)
    assert isinstance(small.surprisal_cdf(mu), float)
    assert isinstance(cal.pvalue(live, mu), float)
    sp = small.saddlepoint(mu)
    for key in ("x", "pdf", "cdf", "sf", "t"):
        assert isinstance(sp[key], float), key
    assert isinstance(sp["converged"], bool)

    for value in ([mu], np.array([mu]), np.array([mu, mu + 1.0])):
        assert isinstance(small.surprisal_pdf(value), np.ndarray)
        assert isinstance(small.surprisal_cdf(value), np.ndarray)
        assert isinstance(cal.pvalue(live, value), np.ndarray)
        assert isinstance(small.saddlepoint(value)["cdf"], np.ndarray)

    assert small.surprisal_pdf(np.array([mu, mu + 1.0])).shape == (2,)
    assert small.saddlepoint(np.array([mu, mu + 1.0]))["cdf"].shape == (2,)


def test_pvalue_never_exceeds_one(small):
    """The survival function is a normalised cumulative sum, so its first
    entry lands a few ulps above 1. A p-value may not."""
    cal = small.calibrate(bins=64)
    for L in cal.lengths:
        assert cal.pvalue(L, -1e6) <= 1.0
        assert cal.pvalue(L, 1e12) >= 0.0
        assert float(np.max(cal.pvalue(L, np.array([-1e6, 0.0])))) <= 1.0


# ═══════════════════════════════════════════════════════════════
# Cross-model summary names
# ════════════════════════════════════════════════════════════════════════


def test_pgen_diagnostics_measures_properness(model):
    d = model.pgen_diagnostics()
    assert d["is_proper"] is True
    assert d["total_mass"] == pytest.approx(1.0, rel=1e-12)
    assert d["length_prior_mass"] == pytest.approx(1.0, rel=1e-12)
    assert d["measure"] == "Pseq"


def test_diversity_profile_carries_entropy_nats(model):
    d = model.diversity_profile()
    assert d["entropy_nats"] == pytest.approx(model.entropy(), rel=1e-12)
    assert d["D1"] == pytest.approx(math.exp(d["entropy_nats"]), rel=1e-9)


def test_length_distribution_is_the_prior(model):
    ld = model.length_distribution
    assert tuple(sorted(ld)) == model.observed_lengths
    assert sum(ld.values()) == pytest.approx(1.0, rel=1e-12)
    assert min(ld) >= 1 and max(ld) <= model.config["max_length"]


def test_n_sequences_is_the_ingested_weight(corpus):
    seqs, abund = corpus
    m = BeamModel(seqs, abundances=abund, order=2, max_length=20)
    assert m.n_sequences == pytest.approx(float(sum(abund)))
    assert m.variant == "beam"


def test_calibration_median_converges_with_the_grid(small):
    """The reconstruction is the only approximation in the null, so refining
    the grid must move its median toward the model's own draws. A biased
    quantile inversion would leave a floor instead: before the
    mid-distribution correction this bias was half a bin wide and did not.
    """
    drawn = small.sample(6000, seed=17)
    u = -small.log_pseq_batch(drawn)
    lengths = np.array([len(s) for s in drawn])
    L = int(np.bincount(lengths).argmax())

    # Convergence is measured against a much finer grid, not against the
    # sample median, because the sample median can be the noisier of the
    # two. It was on the fixture this was written for: about 780 draws gave
    # it a standard error near 0.15 nats against a grid error of 0.012 at
    # 1024 bins, and at 256 bins the estimate came out CLOSER than at 1024,
    # by luck, which is what a reference noisier than the effect will do.
    # (Under the prior this length's surprisal sits on few values and the
    # sample median is good to ~0.001; the grid reference does not care.)
    #
    # 16384 bins is four times finer than the finest grid checked, and its
    # own error is well under the 0.01 asserted below: 0.0014 measured
    # against 65536 bins, which cost twice as long again.
    ref = small.calibrate(bins=16384).median(L)
    medians = {b: small.calibrate(bins=b).median(L)
               for b in (64, 256, 1024, 4096)}
    errors = [abs(m - ref) for m in medians.values()]
    # The coarse grid is decisively worse and the fine one has converged.
    # Monotonicity across every step is NOT asserted, because beyond about
    # 256 bins the remaining differences are where a bin boundary happens to
    # fall relative to the median, which is not ordered in the grid.
    assert errors[0] > 10 * errors[-1], errors
    assert errors[-1] < 0.01, errors

    # And the estimate is close to the drawn sample, within that sample's own
    # resolution rather than within the grid's.
    truth = float(np.median(u[lengths == L]))
    assert abs(medians[1024] - truth) < 0.3


# ════════════════════════════════════════════════════════════════════════


def test_c_and_python_defaults_describe_the_same_model():
    """The two entry points must agree on what "default" means.

    A caller writing BeamModel(seqs) and a caller reaching
    beam_model_config_default() in C are asking for the same model. Once
    already they were not: each layer kept its own copy of the smoothing
    strengths, the copies drifted, and the same corpus gave a different
    model with nothing failing, because both configurations were valid.

    The prior's strengths are the numbers still kept in both layers --
    Python restates BEAM_DEFAULT_ALPHA and BEAM_DEFAULT_ALPHA_LENGTH rather
    than reading them -- so they are held here to the header they copy.
    Order and max_length are served from the C descriptor and checked in
    test_nucleotide.py. Storage no longer differs: both layers default to
    sparse, since a dense model under the prior materialises every row.
    """
    import re
    from pathlib import Path

    import BEAM
    from BEAM.model import config as py_config

    header = (Path(BEAM.__file__).resolve().parents[2]
              / "include" / "beam" / "model.h")
    if not header.is_file():                  # installed, not a source tree
        pytest.skip("no C headers beside the package")
    text = header.read_text()
    c_alpha, c_alpha_length = (
        float(re.search(rf"#define\s+{name}\s+(\S+)", text).group(1))
        for name in ("BEAM_DEFAULT_ALPHA", "BEAM_DEFAULT_ALPHA_LENGTH"))
    assert py_config.DEFAULT_ALPHA == c_alpha
    assert py_config.DEFAULT_ALPHA_LENGTH == c_alpha_length

    cfg = BeamModel(["CASSF", "CASTF", "CAGSF"], abundances=[3, 2, 1],
                    order=2, max_length=6).config
    assert cfg["alpha"] == (c_alpha,) * 3
    assert cfg["alpha_length"] == c_alpha_length
    assert cfg["sparse"] is True
    assert cfg["weights"] == (0.5, 0.5)

    # And the C layer keeps what it is handed, field for field: a strength
    # dropped or reordered at the boundary would build a valid other model.
    handed = {**cfg, "alpha": (1.5, 2.5, 3.5), "alpha_length": 4.5}
    back = _c.config(_c.new(*_config_args_from_dict(handed)))
    for field in ("order", "max_length", "weights", "alphabet", "alpha",
                  "alpha_length", "sparse"):
        assert back[field] == handed[field], field
