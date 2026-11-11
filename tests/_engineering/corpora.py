"""The corpora every suite draws on, and the constants that go with them.

Was `_helpers.py` inside the suites, imported as a bare `_helpers` because
`tests/conftest.py` put that directory on `sys.path`. Nothing about the name
said which layer it belonged to or who was allowed to use it; both are now in
the import line.

Nothing here computes an expected value. These are inputs.
"""
from __future__ import annotations

import random

AA = "ACDEFGHIKLMNPQRSTVWY"
J_MOTIFS = ("EQYF", "EQFF", "NTIYF", "ETQYF", "YGYTF")


def make_corpus(n=1500, seed=11, lo=11, hi=18):
    """CDR3-shaped strings of EXACTLY the requested length.

    A conserved start, a random middle, and a J-like motif at the end, which
    is the structure the two anchors are supposed to pick up. Sequences are
    built to the target length rather than to a fixed prefix plus motif, so
    that short-length fixtures really are short: getting that wrong silently
    turns a length-3 test into a length-8 one.
    """
    rng = random.Random(seed)
    seqs, abund = [], []
    for _ in range(n):
        length = rng.randint(lo, hi)
        motifs = [m for m in J_MOTIFS if len(m) + 5 <= length]
        j = rng.choice(motifs) if motifs else "F"
        head = "CASS" if length >= len(j) + 5 else "C"
        fill = length - len(head) - len(j)
        mid = "".join(rng.choice(AA) for _ in range(fill))
        seq = head + mid + j
        assert len(seq) == length, (seq, length)
        seqs.append(seq)
        abund.append(rng.randint(1, 40))
    return seqs, abund


NT = "ACGT"
NT_J_MOTIFS = ("GAGCAGTACTTC", "GAGCAGTTCTTC", "AATACCATCTAT",
               "GAAACGCAGTAT", "TACGGGTATACC")


def make_nt_corpus(n=1200, seed=17, lo=21, hi=33):
    """The nucleotide counterpart of :func:`make_corpus`.

    Same shape and the same reason for it: a conserved start, a random middle
    and a J-like motif, so the two anchors have something to pick up. Lengths
    are multiples of three only incidentally -- the model knows nothing about
    reading frame, and building the corpus as though it did would hide a
    packing error that happens to respect codon boundaries.
    """
    rng = random.Random(seed)
    seqs, abund = [], []
    for _ in range(n):
        length = rng.randint(lo, hi)
        motifs = [m for m in NT_J_MOTIFS if len(m) + 6 <= length]
        j = rng.choice(motifs) if motifs else "TTC"
        head = "TGTGCC" if length >= len(j) + 6 else "TGT"
        fill = length - len(head) - len(j)
        mid = "".join(rng.choice(NT) for _ in range(fill))
        seq = head + mid + j
        assert len(seq) == length, (seq, length)
        seqs.append(seq)
        abund.append(rng.randint(1, 40))
    return seqs, abund


#: below every tolerance asserted against it.
ANALYSIS_BINS = 384

#: reconstructed null is still a few hundredths of an IQR out. Still ~2s.
CALIBRATION_BINS = 1024


def reads_400(seed=400, n=40, length=400):
    """`n` nucleotide reads of exactly `length` bases, seeded.

    Defined once. `test_persistence.py` and `test_configuration.py` each held
    the same four lines with the same seed -- 40 reads of 400 bases from
    `random.Random(400)` -- one to check a save/load round trip at length 400
    and one to check the threaded path there. Two copies of a corpus is two
    places to change when the long-model path needs different data, and no
    way to tell from either that the other exists.
    """
    rng = random.Random(seed)
    return ["".join(rng.choice("ACGT") for _ in range(length))
            for _ in range(n)]
