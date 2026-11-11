"""Built fixtures, and the one place that says where the files are.

A fixture here is an input that costs something to build: a model, or a model
plus the enumeration of its whole space. The two enumerated models were
copied verbatim into `test_mellin_line.py` and `test_mellin_complex.py` --
same seeds, same corpora, same `log_pseq_batch` over every string, twice --
which is two models built where one would do, and two places to edit when the
enumeration has to change.

Nothing here computes an expected value. The references are in
`tests/oracles/`.
"""
from __future__ import annotations

import itertools
import random
from pathlib import Path

import numpy as np

from BEAM import BeamModel

#: Every amino-acid residue, in the order the model reads them.
AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"

#: The nucleotide alphabet INCLUDING N, which is a residue and not a gap: the
#: enumerated support is every string over all five.
NT_ALPHABET = "ACGTN"


def fixtures_dir() -> Path:
    """`tests/fixtures/`, found from this file and not from the caller's.

    `test_format_compatibility.py` reached it as
    `Path(__file__).resolve().parents[2] / "fixtures"`, which means "two
    directories up from wherever this file happens to sit" and so changes
    meaning the moment the file moves -- and moving files is what a layer
    structure is. The two `.beam` files it points at are real output from
    builds that no longer exist and cannot be regenerated.
    """
    return Path(__file__).resolve().parents[1] / "fixtures"


def enumerated_nt_model():
    """A nucleotide model of `max_length` 6, and `log P` for every string.

    5 residues to length 6 is 19,530 strings: small enough to hold the whole
    space in a list, which is what makes a sum over it an independent check on
    a transform of it.
    """
    rng = random.Random(5)
    seqs = ["TG" + "".join(rng.choice("ACGT") for _ in range(rng.randint(0, 3)))
            + "C" for _ in range(400)]
    abund = [rng.randint(1, 40) for _ in seqs]
    m = BeamModel(seqs, abundances=abund, alphabet="nt", order=3, max_length=6)
    every = ["".join(t) for L in range(1, 7)
             for t in itertools.product(NT_ALPHABET, repeat=L)]
    return m, np.asarray(m.log_pseq_batch(every), dtype=np.float64)


def enumerated_aa_model():
    """The amino-acid counterpart: `max_length` 3, so 20 + 400 + 8,000."""
    rng = random.Random(11)
    seqs = ["C" + "".join(rng.choice("ASGLT") for _ in range(rng.randint(0, 1)))
            + "F" for _ in range(300)]
    abund = [rng.randint(1, 9) for _ in seqs]
    m = BeamModel(seqs, abundances=abund, order=2, max_length=3)
    every = ["".join(t) for L in range(1, 4)
             for t in itertools.product(AA_ALPHABET, repeat=L)]
    return m, np.asarray(m.log_pseq_batch(every), dtype=np.float64)
