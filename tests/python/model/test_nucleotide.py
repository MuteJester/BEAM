"""The nucleotide alphabet, through the Python surface.

The C suite already checks the model itself -- that the partition function
normalises at radix 6, that path counts match enumeration. What is checked
here is the layer on top: that `alphabet="nt"` reaches the C config intact,
that the defaults the two layers resolve are the same defaults, and that
nothing about the amino-acid path moved.
"""
import math

import numpy as np
import pytest

import BEAM
from BEAM.model import alphabets


# Real TCR-beta CDR3 junctions, 36 and 39 bases.
NT_SEQS = [
    "TGTGCCAGCAGTGATAGCGGGAGCTACGAGCAGTACTTC",
    "TGTGCCAGCAGTGATAGCGGGAGCTACGAGCAGTTCTTC",
    "TGTGCCAGCAGCCCGGACAGGGCCTACAATACCATCTAT",
    "TGTGCCAGCAGCTTAGCGGGAGGGAACACCATATATTTT",
    "TGTGCCAGCAGTTACTCTAGCGGGAGCTACGAGCAGTAC",
]
NT_AB = [8.0, 5.0, 3.0, 2.0, 1.0]


@pytest.fixture(scope="module")
def nt():
    """Small enough to build fast; the defaults are exercised separately."""
    return BEAM.BeamModel(NT_SEQS, abundances=NT_AB, alphabet="nt",
                          order=3, max_length=42)


# ═══════════════════════════════════════════════════════════════
# the registry
# ═══════════════════════════════════════════════════════════════

def test_the_registry_is_served_from_the_c_descriptor():
    """Not a second copy of the numbers.

    config.c records a real bug from the last time each layer kept its own
    defaults: they disagreed, silently, and both produced a model. Python must
    read them rather than restate them.
    """
    assert alphabets.names() == ("aa", "nt")
    aa, nt_ = alphabets.info("aa"), alphabets.info("nt")
    assert aa["residues"] == 20 and aa["radix"] == 21
    assert nt_["residues"] == 5 and nt_["radix"] == 6
    assert nt_["symbols"] == "ACGTN"


def test_an_unknown_alphabet_is_refused_by_name():
    with pytest.raises(ValueError, match="unknown alphabet"):
        BEAM.BeamModel(["CASSF"], alphabet="protein")
    with pytest.raises(ValueError, match="unknown alphabet"):
        alphabets.info("rna")


@pytest.mark.parametrize("name", ["aa", "nt"])
def test_python_and_c_resolve_the_same_defaults(name):
    """The property config.c warns about, checked directly.

    A model built with everything left to default must report exactly the
    descriptor's defaults back through `config`, so there is no room for the
    Python layer to have resolved something else.
    """
    info = alphabets.info(name)
    seqs = ["CASSLGQAYEQYF"] if name == "aa" else [NT_SEQS[0]]
    m = BEAM.BeamModel(seqs, alphabet=name)
    cfg = m.config
    assert cfg["alphabet"] == name
    assert cfg["order"] == info["default_order"]
    assert cfg["max_length"] == info["default_max_length"]


def test_the_amino_acid_defaults_did_not_move():
    """Making order and max_length alphabet-resolved must not have
    changed what an amino-acid caller gets."""
    cfg = BEAM.BeamModel(["CASSLGQAYEQYF"]).config
    assert (cfg["alphabet"], cfg["order"],
            cfg["max_length"]) == ("aa", 4, 27)


# ═══════════════════════════════════════════════════════════════
# building and scoring
# ═══════════════════════════════════════════════════════════════

def test_a_nucleotide_model_trains_and_scores(nt):
    assert nt.alphabet == "nt"
    assert nt.n_residues == 5
    assert nt.n_observed == len(NT_SEQS)
    assert nt.n_skipped == 0
    for s in NT_SEQS:
        assert math.isfinite(nt.log_pseq(s))


def test_case_is_folded_end_to_end(nt):
    """Bit-identical, not close: the fold happens in the decode table, so the
    two encode to the same indices and take the same path."""
    for s in NT_SEQS:
        assert nt.log_pseq(s.lower()) == nt.log_pseq(s)
    mixed = "".join(c.lower() if i % 2 else c
                    for i, c in enumerate(NT_SEQS[0]))
    assert nt.log_pseq(mixed) == nt.log_pseq(NT_SEQS[0])


def test_N_is_a_residue_and_other_iupac_codes_are_not(nt):
    """N says "this base is unknown", which the model can represent. R says
    "A or G", which it cannot."""
    with_n = NT_SEQS[0][:-1] + "N"
    assert nt.log_pseq(with_n) <= 0.0        # scoreable, finite or -inf
    with pytest.raises(ValueError, match="alphabet does not accept"):
        nt.explain(NT_SEQS[0][:-1] + "R")


def test_an_amino_acid_sequence_is_refused_by_a_nucleotide_model(nt):
    with pytest.raises(ValueError, match="alphabet does not accept"):
        nt.explain("CASSLGQAYEQYF")


def test_the_refusal_names_the_alphabet_it_belongs_to(nt):
    aa = BEAM.BeamModel(["CASSLGQAYEQYF"], order=2, max_length=15)
    with pytest.raises(ValueError, match=r"'aa' alphabet"):
        aa.explain("CASSXQYF")
    with pytest.raises(ValueError, match=r"'nt' alphabet"):
        nt.explain("ACGTZ")


# ═══════════════════════════════════════════════════════════════
# the analytics, over a nucleotide trellis
# ═══════════════════════════════════════════════════════════════

def test_the_analytics_run_and_agree_with_each_other(nt):
    """Entropy is bounded above by the log of the support, which is the
    cheapest statement that ties the spectrum sweep and the path count to the
    same trellis rather than to two."""
    support = nt.path_count()
    assert support > 0
    h = nt.entropy()
    assert 0.0 <= h <= math.log(support) + 1e-9
    assert nt.hill_number(2.0) >= 1.0
    assert math.isfinite(nt.hill_number(1.0))


def test_generation_emits_only_nucleotides(nt):
    drawn = nt.sample(200, seed=11)
    assert len(drawn) == 200
    assert set("".join(drawn)) <= set("ACGTN")


def test_top_k_returns_scoreable_sequences(nt):
    top = nt.top_k_sequences(10)
    assert len(top) > 0
    for s, lp in top:
        assert set(s) <= set("ACGTN")
        assert nt.log_pseq(s) == pytest.approx(lp, rel=1e-12)


# ═══════════════════════════════════════════════════════════════
# persistence and algebra
# ═══════════════════════════════════════════════════════════════

def test_a_nucleotide_model_round_trips_through_a_file(nt, tmp_path):
    path = tmp_path / "nt_model"
    nt.save(str(path))
    back = BEAM.BeamModel.from_file(str(path) + ".beam")
    assert back.alphabet == "nt"
    assert back.n_residues == 5
    for s in NT_SEQS:
        assert back.log_pseq(s) == nt.log_pseq(s)


def test_the_two_alphabets_cannot_be_combined(nt):
    aa = BEAM.BeamModel(["CASSLGQAYEQYF", "CASSLGQAYEQFF"],
                        order=3, max_length=42)
    # Same order, max_length and depth_cap: everything the shape check
    # compared before the alphabet joined it.
    assert aa.config["order"] == nt.config["order"]
    assert aa.config["max_length"] == nt.config["max_length"]
    with pytest.raises(Exception):
        aa.add(nt)


# ═══════════════════════════════════════════════════════════════
# streaming and tabular entry points
# ═══════════════════════════════════════════════════════════════

def test_the_builder_takes_the_alphabet_too():
    """A streamed build and a one-shot build must produce the same model, and
    that claim is only as good as the config both are handed."""
    b = BEAM.BeamModel.builder(alphabet="nt", order=3, max_length=42)
    b.observe(NT_SEQS[:3], NT_AB[:3])
    b.observe(NT_SEQS[3:], NT_AB[3:])
    streamed = b.build()

    one_shot = BEAM.BeamModel(NT_SEQS, abundances=NT_AB, alphabet="nt",
                              order=3, max_length=42)
    assert streamed.alphabet == "nt"
    for s in NT_SEQS:
        assert streamed.log_pseq(s) == one_shot.log_pseq(s)


def test_the_sequence_column_defaults_to_the_alphabets_own():
    """`cdr3` is the AIRR nucleotide junction and `cdr3_aa` its translation.
    Reading the wrong one skips every row, which is silent until the corpus
    comes back empty."""
    frame = {"cdr3": NT_SEQS, "cdr3_aa": ["CASSLGQAYEQYF"] * len(NT_SEQS),
             "count": NT_AB}
    m = BEAM.BeamModel.from_dataframe(frame, abundance_column="count",
                                      alphabet="nt", order=3, max_length=42)
    assert m.alphabet == "nt"
    assert m.n_observed == len(NT_SEQS)

    aa = BEAM.BeamModel.from_dataframe(frame, abundance_column="count",
                                       order=2, max_length=15)
    assert aa.alphabet == "aa"
    assert aa.n_observed == len(NT_SEQS)


# ═══════════════════════════════════════════════════════════════
# the amino-acid path, unmoved
# ═══════════════════════════════════════════════════════════════

def test_an_amino_acid_model_is_unaffected_by_a_nucleotide_one(corpus):
    """Built either side of a nucleotide model, an amino-acid model must score
    identically -- the cheapest check that no descriptor or dimension leaked
    between them."""
    seqs, ab = corpus
    before = BEAM.BeamModel(seqs, abundances=ab, order=3, max_length=15)
    scores_before = [before.log_pseq(s) for s in seqs[:20]]

    BEAM.BeamModel(NT_SEQS, abundances=NT_AB, alphabet="nt",
                   order=3, max_length=42)

    after = BEAM.BeamModel(seqs, abundances=ab, order=3, max_length=15)
    assert [after.log_pseq(s) for s in seqs[:20]] == scores_before


# ═══════════════════════════════════════════════════════════════
# the open world, at the alphabet's own default order
# ═══════════════════════════════════════════════════════════════

def test_the_open_world_builds_at_the_default_order():
    """The exact configuration that segfaulted.

    `strict=False` turns on every backoff strength, and the nucleotide default
    order is 6. The wide expert's backoff base was indexed at `order - 1` into
    an array holding the mid hierarchy's orders 0 to 3 -- in range for every
    amino-acid order, two past the end here. Nothing in the suite built a
    nucleotide model with backoff on, so nothing saw it.
    """
    m = BEAM.BeamModel(NT_SEQS, abundances=NT_AB, alphabet="nt", strict=False)
    assert m.config["order"] == 6
    for s in NT_SEQS:
        assert math.isfinite(m.log_pseq(s))
    # The open world gives mass to sequences the corpus never spelled; that is
    # what it is for, and it is the property the backoff base carries. The
    # probe has to be of an OBSERVED length, or the length prior rules it out
    # before any residue is looked at and the test proves nothing.
    unseen = "A" * len(NT_SEQS[0])
    assert unseen not in NT_SEQS
    assert math.isfinite(m.log_pseq(unseen))
    # And the closed world must still refuse it, or "open" means nothing.
    closed = BEAM.BeamModel(NT_SEQS, abundances=NT_AB, alphabet="nt")
    assert closed.log_pseq(unseen) == -math.inf


@pytest.mark.parametrize("order", [4, 5, 6, 7])
def test_every_order_survives_backoff(order):
    """Order 4 is the boundary the base cap leaves alone; 5 upward exercise it.

    Stops at 7 because backoff forces dense storage, and a dense nucleotide
    wide table is `(depth_cap + 1) * 6**order * 5` doubles per direction --
    0.3 GB at order 7 and 11 GB at order 9. Orders 5, 6 and 7 take the clamped
    branch through identical code; the ceiling is built separately below,
    where the closed world makes it affordable.
    """
    m = BEAM.BeamModel(NT_SEQS, abundances=NT_AB, alphabet="nt", order=order,
                       max_length=42, strict=False)
    assert math.isfinite(m.log_pseq(NT_SEQS[0]))
    assert m.path_count() > 0


def test_a_smoothed_nucleotide_model_is_still_a_distribution():
    """Changing which table the wide expert backs off to could leave a model
    that builds and scores but no longer sums to 1."""
    m = BEAM.BeamModel(["TGT", "TGC", "GCC", "TCA"], abundances=[5, 3, 2, 4],
                       alphabet="nt", order=5, max_length=5,
                       strict=False)
    logpi = m.log_length_prior(3)
    total = 0.0
    for a in "ACGTN":
        for b in "ACGTN":
            for c in "ACGTN":
                lp = m.log_pseq(a + b + c)
                if math.isfinite(lp):
                    total += math.exp(lp - logpi)
    assert total == pytest.approx(1.0, abs=1e-9)


def test_the_ceiling_order_builds():
    """Order 9 built, not merely asserted legal.

    That distinction is the whole lesson of the defect this file's other tests
    cover: the suite asserted `beam_config_valid` accepted order 9 and never
    constructed one, and construction is where it segfaulted. Closed world, so
    storage is sparse and this costs a few hundred megabytes rather than
    eleven gigabytes.
    """
    m = BEAM.BeamModel(["ACGTAC", "ACGTAG"], abundances=[2.0, 1.0],
                       alphabet="nt", order=9, max_length=6)
    assert m.config["order"] == 9
    assert m.n_states == 6 ** 9
    assert math.isfinite(m.log_pseq("ACGTAC"))


# ═══════════════════════════════════════════════════════════════
# the memory guard
# ═══════════════════════════════════════════════════════════════

def test_a_configuration_larger_than_the_budget_is_refused():
    """Refused before allocating, not after.

    calloc grants an absurd size lazily on Linux, so without this guard the
    process does not fail where it asked -- it fails when the tables are first
    written, with the OOM killer choosing a victim from the whole machine.
    Lowering the budget is how the refusal is reached without allocating
    anything large to reach it.
    """
    before = BEAM.memory_budget()
    try:
        BEAM.set_memory_budget(1_000_000)          # 1 MB
        with pytest.raises(MemoryError, match="memory budget"):
            BEAM.BeamModel(NT_SEQS, abundances=NT_AB, alphabet="nt",
                           order=6, max_length=42, strict=False)
    finally:
        BEAM.set_memory_budget(before)
    assert BEAM.memory_budget() == before
    # and the same model builds once the budget allows it
    m = BEAM.BeamModel(NT_SEQS, abundances=NT_AB, alphabet="nt",
                       order=6, max_length=42, strict=False)
    assert m.alphabet == "nt"


def test_the_refusal_says_what_it_needs_and_what_it_has():
    """A bare MemoryError leaves a caller nothing to act on."""
    before = BEAM.memory_budget()
    try:
        BEAM.set_memory_budget(1_000_000)
        with pytest.raises(MemoryError) as exc:
            BEAM.BeamModel(NT_SEQS, alphabet="nt", strict=False)
    finally:
        BEAM.set_memory_budget(before)
    text = str(exc.value)
    assert "needs about" in text and "memory budget is" in text
    assert "set_memory_budget" in text        # names the way out
