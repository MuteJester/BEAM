"""The stand-in repertoires the documentation's group examples are built from.

They are not real data, so the examples are only honest if the built-in
differences between stand-in populations are what the pages say they are.
These tests pin that, with a small fake reference in place of BEAM's.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples"))
import _standins as st  # noqa: E402

AA = "ACDEFGHIKLMNPQRSTVWY"


class FakeReference:
    """Anything with `sample(n, seed=)` will do: random CDR3-like strings."""

    def sample(self, n, seed=None):
        rng = np.random.default_rng(seed)
        out = []
        for _ in range(n):
            length = int(rng.integers(10, 19))
            middle = "".join(rng.choice(list(AA), size=length - 7))
            out.append("CASS" + middle + "QYF")
        return out


@pytest.fixture
def ref():
    return FakeReference()


def test_a_population_is_reproducible(ref):
    assert st.population(ref, "B", 2000, seed=3) == st.population(ref, "B", 2000, seed=3)


def test_population_b_has_longer_sequences_than_a(ref):
    a = st.population(ref, "A", 5000, seed=1)
    b = st.population(ref, "B", 5000, seed=1)
    assert np.mean([len(s) for s in b]) > np.mean([len(s) for s in a]) + 1.0


def test_population_c_carries_the_motif_more_often_than_a(ref):
    a = st.population(ref, "A", 5000, seed=1)
    c = st.population(ref, "C", 5000, seed=1)
    share = lambda cells: np.mean([st.has_motif(s) for s in cells])  # noqa: E731
    assert share(c) > 3 * share(a)


def test_population_returns_the_requested_number_of_cells(ref):
    assert len(st.population(ref, "C", 1234, seed=2)) == 1234


def test_an_unknown_population_is_refused(ref):
    with pytest.raises(ValueError, match="population"):
        st.population(ref, "Z", 10, seed=1)


def test_no_expansion_keeps_the_sampled_counts(ref):
    cells = ["CASSA", "CASSA", "CASSB"]
    seqs, counts = st.expand(cells, seed=1, strength=0)
    assert dict(zip(seqs, counts)) == {"CASSA": 2, "CASSB": 1}


def test_expansion_creates_large_clones_reproducibly(ref):
    cells = st.population(ref, "A", 3000, seed=4)
    seqs0, counts0 = st.expand(cells, seed=5, strength=0)
    seqs2, counts2 = st.expand(cells, seed=5, strength=2.0)
    assert seqs0 == seqs2                         # the same clones, larger
    assert max(counts2) > 20 * max(counts0)
    assert sum(counts2) > sum(counts0)
    assert st.expand(cells, seed=5, strength=2.0) == (seqs2, counts2)


def test_donors_of_one_group_differ_but_share_the_population(ref):
    d1 = st.population(ref, "B", 3000, seed=10)
    d2 = st.population(ref, "B", 3000, seed=11)
    assert d1 != d2
    assert abs(np.mean([len(s) for s in d1]) - np.mean([len(s) for s in d2])) < 0.3
