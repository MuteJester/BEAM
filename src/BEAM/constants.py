"""Fixed dimensions of the model, mirrored from the C headers.
"""
from __future__ import annotations



from ._simulation_result import SimulationResult


BEAM_RESIDUES = 20

#: The amino-acid default sequence length. Mirrors ``BEAM_MAX_LENGTH``.
#: This was the ceiling too while there was one alphabet; the ceiling is now
#: ``BEAM_MAX_LENGTH_CEILING`` and the default comes from the alphabet.
BEAM_MAX_LENGTH = 27

#: The structural ceiling on ``max_length``, mirroring
#: ``BEAM_MAX_LENGTH_CEILING``. The field is a ``uint8_t`` in C, so 255 is the
#: most any alphabet could ask for. Nucleotide sequences are three times the
#: length of the amino acids they encode, so 27 cannot bound both.
BEAM_MAX_LENGTH_CEILING = 255

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
