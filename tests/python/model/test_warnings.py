"""The performance warning channel."""
from __future__ import annotations

import random
import warnings

import pytest

import BEAM
from BEAM import BeamModel

SHORT = ["CASSLGEQYF", "CASSQDTQYF", "CASSLAPGATNEKLFF", "CASRDRGNTEAFF"]


@pytest.fixture(scope="module")
def reads_400():
    rng = random.Random(400)
    return ["".join(rng.choice("ACGT") for _ in range(400)) for _ in range(40)]


def test_a_long_model_warns_about_the_quadratic_before_building(reads_400):
    with pytest.warns(BEAM.BeamPerformanceWarning, match="max_length") as rec:
        BeamModel(reads_400[:20], order=2, max_length=400, alphabet="nt")
    text = " ".join(str(w.message) for w in rec)
    assert "24.1x" in text and "81" in text
    assert rec[0].filename == __file__


def test_a_default_length_model_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error", BEAM.BeamPerformanceWarning)
        BeamModel(SHORT, order=2)


def test_data_length_clamps_at_the_cap_and_counts_what_it_dropped(reads_400):
    reads = reads_400 + ["ACGT" * 150]          # one 600-base read
    # Both warnings fire, so capture the record and assert over it. Nesting a
    # match= block per warning only passes on pytest 8+, which re-emits the
    # unmatched one into the outer block; the project floor is pytest>=7.
    with pytest.warns(BEAM.BeamPerformanceWarning) as rec:
        m = BeamModel(reads, max_length="data", order=2, alphabet="nt")
    said = " ".join(str(w.message) for w in rec)
    assert "max_length=512" in said and "skipped" in said, said
    assert m.config["max_length"] == 512 and m.n_skipped == 1


@pytest.mark.parametrize("make", ["builder", "from_sequence_file", "from_file"])
def test_every_way_to_reach_a_long_model_warns(make, reads_400, tmp_path):
    """The channel must have no hole. Streaming is the documented path for a
    corpus too large for memory -- the case this length ceiling exists for --
    and a load re-finalizes from the stored counts, so it pays the same cost
    with the caller least likely to know the model's max_length."""
    if make == "builder":
        with pytest.warns(BEAM.BeamPerformanceWarning, match="max_length=400"):
            BeamModel.builder(alphabet="nt", order=2, max_length=400)
        return
    if make == "from_sequence_file":
        path = tmp_path / "reads.txt"
        path.write_text("\n".join(reads_400[:5]) + "\n", encoding="utf-8")
        with pytest.warns(BEAM.BeamPerformanceWarning, match="max_length=400"):
            BeamModel.from_sequence_file(path, alphabet="nt", order=2,
                                         max_length=400)
        return
    with pytest.warns(BEAM.BeamPerformanceWarning):
        m = BeamModel(reads_400[:10], order=2, max_length=400, alphabet="nt")
    saved = tmp_path / "m.beam"
    m.save(saved)
    with pytest.warns(BEAM.BeamPerformanceWarning, match="max_length=400"):
        BeamModel.from_file(saved)


def test_a_rebuilt_model_does_not_re_warn(reads_400):
    """`without` reconstructs `other` from the receiver's own config. Warning
    there would report a max_length the caller never chose."""
    with pytest.warns(BEAM.BeamPerformanceWarning):
        m = BeamModel(reads_400[:10], order=2, max_length=400, alphabet="nt")
    with warnings.catch_warnings():
        warnings.simplefilter("error", BEAM.BeamPerformanceWarning)
        m.without(reads_400[:2])


def test_the_warning_points_at_the_callers_own_line(reads_400):
    """A stacklevel that lands in facade.py tells the user nothing. Every entry
    point one or more library frames out has to add them."""
    pd = pytest.importorskip("pandas")
    table = pd.DataFrame({"junction": reads_400[:5], "count": 1})
    with pytest.warns(BEAM.BeamPerformanceWarning) as rec:
        BeamModel.from_dataframe(table, sequence_column="junction",
                                 abundance_column="count", alphabet="nt",
                                 order=2, max_length=400)
    assert rec[0].filename == __file__, rec[0].filename


def test_the_warning_is_a_user_warning():
    assert issubclass(BEAM.BeamPerformanceWarning, UserWarning)
