"""What a model reports about its own construction.

Three of these are stored and the rest derived, and the split matters: a
stored figure has to survive a round trip, and a derived one has to agree
with the thing it is derived from. Both are checked.
"""
import datetime
import math

import pytest

import BEAM
from BEAM import BeamModel

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
    assert after["build"]["observe_seconds"] == before["build"]["observe_seconds"]
    # Loading re-finalizes, which is work this model has had done to it, so
    # the finalize total is allowed to grow and nothing else is.
    assert after["build"]["finalize_seconds"] >= before["build"]["finalize_seconds"]


def test_size_agrees_with_the_accessors(small):
    meta = small.metadata(lengths=False)
    assert meta["size"]["n_cells"] == small.n_cells
    assert meta["size"]["n_nonzero_cells"] == small.n_nonzero_cells
    assert meta["size"]["n_states"] == small.n_states
    assert meta["size"]["memory_bytes"] == small.memory_bytes
    assert meta["size"]["occupancy"] == pytest.approx(
        small.n_nonzero_cells / small.n_cells)


def test_the_possible_row_count_is_the_length_times_the_vocabulary(small):
    """`2 * L * radix ** order`, because a length-L sequence reads forward
    depths 0..L-1 and backward depths 0..L-1, and each depth admits every
    context."""
    meta = small.metadata()
    order = int(meta["config"]["order"])
    for L, e in meta["lengths"].items():
        assert e["rows_possible"] == 2 * L * 21 ** order
        assert e["cells_possible"] == e["rows_possible"] * small.n_residues


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
    for name in ("a", "b"):
        m = BeamModel(seqs, abundances=ab, order=3, max_length=9,
                      strict=True)
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
    assert created == m.metadata()["created_unix"]
    assert observed > 0 and finalized > 0
