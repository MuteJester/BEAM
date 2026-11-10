"""Fixed dimensions of the model, mirrored from the C headers.
"""
from __future__ import annotations



from ._simulation_result import SimulationResult


BEAM_RESIDUES = 20

#: The amino-acid default sequence length. Mirrors ``BEAM_MAX_LENGTH``.
#: This was the ceiling too while there was one alphabet; the ceiling is now
#: ``BEAM_MAX_LENGTH_CEILING`` and the default comes from the alphabet.
BEAM_MAX_LENGTH = 27

#: The cross-alphabet maximum of ``max_length``, mirroring
#: ``BEAM_MAX_LENGTH_CEILING``: what a configuration may ask for, whichever
#: alphabet it uses. The field is a ``uint32_t`` in C. It is not the bound
#: any one model is held to; that is per-alphabet (255 for amino acids, 512
#: for nucleotides) and comes from the descriptor as
#: ``alphabets.info(a)["max_max_length"]``.
BEAM_MAX_LENGTH_CEILING = 512

#: Size of the fixed per-sequence length arrays, mirroring
#: ``BEAM_LENGTH_BUFFER``. This is a storage size, not a validation limit:
#: it says how long a sequence the arrays can hold, where
#: ``BEAM_MAX_LENGTH_CEILING`` says how long a configuration may ask for.
BEAM_LENGTH_BUFFER = 512

#: Largest order any alphabet permits, mirroring ``BEAM_MAX_ORDER``. It is a
#: sizing bound, not the validation limit: what a given model may ask for
#: comes from its own alphabet, since the real constraint is on
#: ``radix ** order``.
BEAM_MAX_ORDER = 9

#: Largest moment order the jet DP carries. Mirrors ``BEAM_MAX_MOMENT``.
BEAM_MAX_MOMENT = 16

#: The twenty amino acids, in table order. Kept under its historical name;
#: :mod:`BEAM.model.alphabets` serves this and the nucleotide symbols from the
#: C descriptor, which is where new code should read them.
BEAM_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"

#: The five nucleotide residues, in table order. ``N`` is a residue rather
#: than a wildcard: a read carrying one is data the model can represent.
BEAM_NT_ALPHABET = "ACGTN"
