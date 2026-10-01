"""What a model reports about its own construction.

Three of these are stored and the rest derived, and the split matters: a
stored figure has to survive a round trip, and a derived one has to agree
with the thing it is derived from. Both are checked.
"""
import datetime
import math

import pytest

import BEAM
from BEAM import BeamModel, Builder

from _helpers import make_corpus


SEQS = ["CASSLGEQYF", "CASSQDTQYF", "CASSLGTEAFF", "CASSPGTGFF"]


@pytest.fixture(scope="module")
def small():
    seqs, abund = make_corpus(n=400, seed=7, lo=9, hi=12)
    return BeamModel(seqs, abundances=abund, order=2, max_length=14)


def test_metadata_reports_a_real_creation_time(small):
    """A date, not an interval, so it comes from the realtime clock."""
    meta = small.metadata(lengths=False)
    stamp = datetime.datetime.fromisoformat(meta["created"])
    now = datetime.datetime.now(datetime.timezone.utc)
    assert stamp <= now
    assert (now - stamp).total_seconds() < 3600
    assert meta["created_unix"] == int(stamp.timestamp())


def test_build_time_aggregates_across_observe_calls():
    """Streaming a build in chunks must report the whole build, not the last
    chunk. That is the reason the counter lives on the model rather than
    being timed at the call site."""
    one = BeamModel(SEQS * 40, order=2, max_length=12)

    many = BeamModel.builder(order=2, max_length=12)
    for _ in range(40):
        many.observe(SEQS)
    many = many.build()

    a = one.metadata(lengths=False)["build"]
    b = many.metadata(lengths=False)["build"]
    assert a["n_observed"] == b["n_observed"]
    # Forty calls cannot have cost less than the one that did the same work
    # in a single pass; the point is that the total is accumulated at all.
    assert b["observe_seconds"] > 0.0
    assert b["total_seconds"] >= b["observe_seconds"]


def test_provenance_survives_a_round_trip(small, tmp_path):
    """A loaded model was built when it was built, not when it was loaded."""
    before = small.metadata(lengths=False)
    p = tmp_path / "prov.beam"
    small.save(str(p))
    after = BeamModel.from_file(str(p)).metadata(lengths=False)

    assert after["created_unix"] == before["created_unix"]
    assert after["build"] == before["build"]

    # Exactly equal, not merely close, and that is the part worth pinning.
    # A `.beam` file stores counts rather than finalized tables, so every load
    # rebuilds them -- and the first version of this charged that finalize to
    # the model, which made the reported build time rise each time the model
    # was opened and bake the rise in if it was ever saved again. The load's
    # own finalize is now excluded; a re-finalize the caller asks for is not.
    assert after["build"]["finalize_seconds"] == before["build"]["finalize_seconds"]


def test_a_caller_s_own_refinalize_still_counts(small, tmp_path):
    """The counterpart to the line above, so the exclusion cannot widen.

    Excluding the load's finalize is right because nobody asked for it. It
    would be wrong to stop counting a finalize that someone did ask for, and
    without this test the cheapest way to make the round trip exact would be
    to stop counting finalizes altogether.
    """
    seqs = ["CASSLGQ", "CASSLGT", "CASRDTQ"]
    m = BeamModel(seqs, abundances=[3.0, 2.0, 1.0], order=2, max_length=9)
    before = m.metadata(lengths=False)["build"]["finalize_seconds"]
    other = BeamModel(["CASSLGY"], abundances=[1.0], order=2, max_length=9)
    m.add(other)
    after = m.metadata(lengths=False)["build"]["finalize_seconds"]
    assert after > before


def test_size_agrees_with_the_accessors(small):
    meta = small.metadata(lengths=False)
    assert meta["size"]["n_cells"] == small.n_cells
    assert meta["size"]["n_nonzero_cells"] == small.n_nonzero_cells
    assert meta["size"]["n_states"] == small.n_states
    assert meta["size"]["memory_bytes"] == small.memory_bytes
    assert meta["size"]["occupancy"] == pytest.approx(
        small.n_nonzero_cells / small.n_cells)


def test_the_possible_row_count_is_what_a_sequence_can_reach(small):
    """`2 * sum_d R ** min(d, order)`, not `2 * L * radix ** order`.

    A length-L sequence reads forward depths 0..L-1 and backward depths
    0..L-1, but a depth does NOT admit every context: at depth `d` only
    `min(d, order)` slots are free, the rest being the sentinel that pads
    that direction's anchor. So `R ** min(d, order)` rows exist there and the
    others cannot be addressed by any sequence at all.

    This was `2 * L * radix ** order`, which counts rows nothing can reach.
    Depth 0 has exactly ONE row, the all-sentinel context, and a model that
    had found it was reported as having found 1/194481 of that depth.
    """
    meta = small.metadata()
    order = int(meta["config"]["order"])
    R = small.n_residues
    for L, e in meta["lengths"].items():
        assert e["rows_possible"] == 2 * sum(R ** min(d, order)
                                             for d in range(L))
        assert e["cells_possible"] == e["rows_possible"] * R
        # And the old formula is strictly looser, which is why it understated.
        assert e["rows_possible"] < 2 * L * 21 ** order


@pytest.mark.parametrize("length", [2, 3, 4])
def test_the_possible_cell_count_is_what_brute_force_occupies(length):
    """Exact against exact, with nothing sampled and nothing assumed.

    Enumerate EVERY sequence of this length over the whole alphabet and the
    cells the model then holds ARE the reachable set. So `cells_possible` must
    equal `n_nonzero_cells`, and coverage must be exactly 1.

    That is the property the old denominator failed: at length 4 it claimed
    1,555,848 possible rows where 16,842 exist, so a model that had seen
    every sequence there is reported at 1.08% instead of at 100%.

    If the cell addressing in `indexing.h` ever changes, this fails rather
    than `metadata()` quietly reporting against a wrong denominator.
    """
    import itertools
    from BEAM.constants import BEAM_ALPHABET as AA
    seqs = ["".join(p) for p in itertools.product(AA, repeat=length)]
    m = BeamModel(seqs, max_length=max(length, 2))
    e = m.metadata()["lengths"][length]

    assert e["cells_possible"] == m.n_nonzero_cells
    assert e["rows_observed"] == e["rows_possible"]
    assert e["cells_observed"] == e["cells_possible"]
    assert e["row_coverage"] == 1.0
    assert e["cell_coverage"] == 1.0


def test_coverage_is_a_fraction_and_rows_bound_cells(small):
    meta = small.metadata()
    for L, e in meta["lengths"].items():
        assert 0.0 <= e["row_coverage"] <= 1.0
        assert 0.0 <= e["cell_coverage"] <= 1.0
        assert e["rows_observed"] <= e["rows_possible"]
        assert e["cells_observed"] <= e["cells_possible"]
        # Every occupied row holds at least one live cell and at most R.
        assert e["rows_observed"] <= e["cells_observed"]
        assert e["cells_observed"] <= e["rows_observed"] * small.n_residues


def test_coverage_grows_with_length_while_the_data_does(small):
    """Depths accumulate, so a longer sequence reads every row a shorter one
    does and then some. The COUNT can only rise with L; the FRACTION need
    not, because the denominator rises too."""
    meta = small.metadata()
    seen = [meta["lengths"][L]["rows_observed"] for L in sorted(meta["lengths"])]
    assert seen == sorted(seen)


def test_shares_sum_to_one(small):
    meta = small.metadata()
    total = sum(e["share"] for e in meta["lengths"].values())
    assert total == pytest.approx(1.0, abs=1e-12)


def test_support_is_off_by_default_and_exact_when_asked(small):
    """`path_count` is a dynamic program per length, so it is opt-in."""
    assert "path_count" not in next(iter(small.metadata()["lengths"].values()))
    meta = small.metadata(support=True)
    for L, e in meta["lengths"].items():
        assert e["path_count"] == small.path_count(L)
        if e["path_count"] > 0:
            assert e["log10_spellable_fraction"] == pytest.approx(
                math.log10(e["path_count"]) - L * math.log10(small.n_residues))


def test_lengths_can_be_skipped(small):
    assert "lengths" not in small.metadata(lengths=False)
    assert "lengths" in small.metadata()


def test_provenance_is_the_only_thing_two_identical_builds_disagree_on(tmp_path):
    """Putting a clock in the header cost a real guarantee, and this is what
    is left of it.

    A `.beam` file used to be a pure function of the corpus: build the same
    model twice, get the same bytes. `check.py` leans on that, hashing a saved
    fixture against a recorded digest, and a C test compares a frozen save
    against a thawed one byte for byte. Provenance breaks it by construction.

    So the format publishes the window it broke, and the property becomes: the
    file still agrees everywhere else. That is strictly weaker than before and
    worth stating plainly -- but it still covers the entire payload, and it is
    checked here rather than assumed by the two callers that skip those bytes.
    """
    seqs = ["CASSLGQ", "CASSLGT", "CASRDTQ", "CGSSLGQ", "CASSLGQY", "CASRDTQY"]
    ab = [40.0, 25.0, 18.0, 12.0, 9.0, 6.0]
    paths = []
    first_created = None
    for name in ("a", "b"):
        m = BeamModel(seqs, abundances=ab, order=3, max_length=9)
        if first_created is None:
            first_created = m.metadata(lengths=False)["created_unix"]
        p = tmp_path / name
        m.save(str(p))
        paths.append(p.with_suffix(".beam").read_bytes())
    a, b = paths
    assert len(a) == len(b)

    off, n = BEAM.file_provenance_span()
    # Everything but the window and the trailing CRC that covers it.
    assert a[:off] == b[:off]
    assert a[off + n:-4] == b[off + n:-4]

    # And the window is aimed at provenance rather than at dead bytes, which
    # is the way this test could pass while skipping the wrong thing. Decoding
    # it says so without depending on two builds taking a different number of
    # nanoseconds, which is true but is not something to assert.
    import struct
    created, observed, finalized = struct.unpack_from("<qqq", a, off)
    # `a` is the FIRST model's bytes, so it must be compared against the first
    # model's own stamp. Comparing against `m`, the loop variable, read the
    # second model's clock and failed whenever the two builds straddled a
    # one-second tick.
    assert created == first_created
    assert observed > 0 and finalized > 0


def test_metadata_describes_a_compacted_model(small, tmp_path):
    """A compacted model is a normal thing to be handed, not an error.

    `compact()` releases the raw counts, and `n_nonzero_cells` inspects them,
    so the first version of this raised and metadata was unavailable on
    exactly the models most worth asking about -- a large one is deployed
    compacted. It now reports None for the figures that genuinely cannot be
    known and everything else as usual.

    The per-length coverage is among the unknowable ones: every finalized
    cell is positive under the prior, so the counts were the only record of
    which cells the corpus reached. What does not depend on them -- the
    ceilings, the shares, Z(L) -- is reported exactly as before.
    """
    seqs, abund = make_corpus(n=300, seed=11, lo=9, hi=12)
    m = BeamModel(seqs, abundances=abund, order=2, max_length=14)
    before = m.metadata()
    m.compact()
    after = m.metadata()

    assert before["size"]["compacted"] is False
    assert after["size"]["compacted"] is True

    # Unknowable after compaction, and said so rather than guessed.
    assert after["size"]["n_nonzero_cells"] is None
    assert after["size"]["occupancy"] is None
    assert before["size"]["n_nonzero_cells"] is not None

    COUNTED = {"rows_observed", "row_coverage", "cells_observed",
               "cell_coverage"}
    assert after["lengths"].keys() == before["lengths"].keys()
    for L, e in after["lengths"].items():
        assert all(e[k] is None for k in COUNTED), L
        assert all(before["lengths"][L][k] is not None for k in COUNTED), L
        assert ({k: v for k, v in e.items() if k not in COUNTED}
                == {k: v for k, v in before["lengths"][L].items()
                    if k not in COUNTED}), L
    assert after["created_unix"] == before["created_unix"]
    assert after["build"] == before["build"]
    assert after["size"]["n_states"] == before["size"]["n_states"]
    assert after["size"]["n_cells"] == before["size"]["n_cells"]


def test_adding_sequences_through_add_carries_their_build_time():
    """`add` is the other way sequences enter a model, so the clock must move.

    The counters aggregated here from the start and the durations did not, so
    a model assembled from shards reported a build time of zero while
    `n_observed` rose. `Builder.snapshot()` goes through this path and was the
    visible symptom: seconds of real counting reported as 0.0.
    """
    a = BeamModel(["CASSLGQ", "CASSLGT"], abundances=[2.0, 1.0],
                  order=2, max_length=9)
    b = BeamModel(["CASSLGY", "CASRDTQ"], abundances=[1.0, 1.0],
                  order=2, max_length=9)
    before = a.metadata(lengths=False)["build"]
    a.add(b)
    after = a.metadata(lengths=False)["build"]

    assert after["n_observed"] == before["n_observed"] + 2
    assert after["observe_seconds"] > before["observe_seconds"]

    # And subtract is its exact inverse, for the same reason the counters are:
    # otherwise a leave-one-out workflow reports a build time that only rises.
    a.subtract(b)
    back = a.metadata(lengths=False)["build"]
    assert back["n_observed"] == before["n_observed"]
    assert back["observe_seconds"] == before["observe_seconds"]


def test_a_streaming_snapshot_reports_the_time_it_spent():
    b = Builder(order=2, max_length=9)
    b.observe(["CASSLGQ", "CASSLGT", "CASRDTQ"], [1.0, 1.0, 1.0])
    meta = b.snapshot().metadata(lengths=False)["build"]
    assert meta["n_observed"] == 3
    assert meta["observe_seconds"] > 0.0


def test_coverage_measures_observation_not_finiteness():
    """Coverage is a question about the corpus, not about the tables.

    This once read the finalized tables and called a cell reached when its
    log probability was not -inf. Under the prior every cell is finite, so a
    model built from two sequences would report that it had discovered the
    WHOLE dictionary -- `row_coverage` 1.0000 -- while `n_nonzero_cells` in
    the same dict said 30. It must count what the corpus reached: the
    longest length reads every depth any shorter one does, so its
    `cells_observed` IS `n_nonzero_cells`, and the fractions stay tiny.
    """
    seqs, ab = ["CASSLGEQYF", "CASSQDTQYF"], [3.0, 1.0]
    m = BeamModel(seqs, abundances=ab, order=2, max_length=12)
    meta = m.metadata()
    per = [e["cells_observed"] for e in meta["lengths"].values()]
    assert max(per) == meta["size"]["n_nonzero_cells"]
    # At the corpus's own length and past it the dictionary is large; the
    # shortest lengths have a handful of rows, which two sequences can fill.
    for L in (10, 11, 12):
        e = meta["lengths"][L]
        assert 0.0 < e["row_coverage"] < 0.01, (L, e["row_coverage"])
        assert 0.0 < e["cell_coverage"] < e["row_coverage"], L


def test_coverage_is_unavailable_on_a_compacted_model():
    """Absent, not saturated. The one case that genuinely cannot be answered.

    `compact()` releases the counts, and under the prior they are the only
    record of which cells the corpus reached: every finalized cell is
    positive, observed or not. So a compacted model reports None rather
    than a coverage of 1.
    """
    seqs, ab = ["CASSLGEQYF", "CASSQDTQYF"], [3.0, 1.0]
    m = BeamModel(seqs, abundances=ab, order=2, max_length=12)
    L = sorted(m.metadata()["lengths"])[0]
    assert m.metadata()["lengths"][L]["rows_observed"] is not None
    m.compact()
    e = m.metadata()["lengths"][L]
    assert e["rows_observed"] is None
    assert e["row_coverage"] is None
    assert e["cells_observed"] is None
    assert e["cell_coverage"] is None
    # The rest of the entry is still exact and still reported.
    assert e["rows_possible"] == 2 * sum(20 ** min(d, 2) for d in range(L))
    assert e["share"] > 0.0


def test_cells_observed_may_not_be_summed_over_lengths():
    """The union over lengths is the LONGEST length, not the sum.

    A cell is addressed by (direction, depth, context) with no length in the
    address, so a depth-3 cell is read by every length from 4 up and the same
    cell is counted once per length. Summing gives 1,907,884 where the model
    holds 429,248 -- more than four times too many -- and the per-length
    ceilings nest the same way, which is why `parameter_discovery` takes the
    union as a max rather than a sum.

    This also cross-checks the frozen-counts branch of
    `beam_model_depth_occupancy`: a sparse model freezes its counts at
    finalize and a dense one does not, so the two reach the figure by
    different code and must still agree exactly. The foundation model is
    sparse, so the frozen branch is the one that runs at scale.
    """
    seqs, abund = make_corpus(n=3000, seed=17, lo=8, hi=12)
    got = {}
    # Order 3 rather than the default 4: the dense twin materialises every
    # row under the prior, which at order 4 is gigabytes and a slow build.
    for sparse in (True, False):
        m = BeamModel(seqs, abundances=abund, order=3, max_length=14,
                      sparse=sparse)
        meta = m.metadata()
        per = {L: e["cells_observed"] for L, e in meta["lengths"].items()}
        assert len(per) > 1, "a single length would make this vacuous"
        assert max(per.values()) == meta["size"]["n_nonzero_cells"]
        assert sum(per.values()) > meta["size"]["n_nonzero_cells"]
        # Longer lengths read every depth a shorter one does, and then some.
        seen = [per[L] for L in sorted(per)]
        assert seen == sorted(seen)
        got[sparse] = per
    assert got[True] == got[False], "frozen and mutable counts must agree"
