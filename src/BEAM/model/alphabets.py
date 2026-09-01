"""The alphabets a model can be built over, and their defaults.

Every number here is read from the C descriptor rather than written down a
second time. That is not tidiness: ``config.c`` records a real bug from the
last time the two layers each kept their own copy of a default. They
disagreed, silently, and both produced a model -- a caller reaching
``beam_model_config_default()`` and a caller writing ``BeamModel(seqs)`` were
asking for the same thing and getting different ones. A second axis has
exactly the same failure mode, so this module resolves nothing on its own.
"""
from __future__ import annotations

from typing import Any

from .. import _cbeam as _c


#: The alphabet used when a caller names none. Amino acid, as it always was.
DEFAULT_ALPHABET = "aa"

#: The column a nucleotide corpus is read from by default. ``cdr3`` is the
#: AIRR field for the nucleotide junction; ``cdr3_aa`` is its translation.
DEFAULT_COLUMN = {"aa": "cdr3_aa", "nt": "cdr3"}


def names() -> tuple[str, ...]:
    """Every alphabet this build knows, in id order."""
    return _c.alphabets()


def info(alphabet: str) -> dict[str, Any]:
    """The C descriptor for `alphabet`.

    Keys: ``name``, ``id``, ``residues``, ``radix``, ``default_order``,
    ``max_order``, ``default_max_length``, ``default_depth_cap``, ``symbols``.
    """
    if not isinstance(alphabet, str):
        raise TypeError(
            f"alphabet must be a string, got {type(alphabet).__name__}")
    try:
        return _c.alphabet_info(alphabet)
    except ValueError:
        raise ValueError(
            f"unknown alphabet {alphabet!r}; this build knows "
            f"{', '.join(repr(n) for n in names())}") from None


def symbols(alphabet: str) -> str:
    """The alphabet's residues, in table order."""
    return info(alphabet)["symbols"]


def accepts(alphabet: str, sequence: str) -> bool:
    """True when every character of `sequence` is a residue of `alphabet`.

    Case-insensitive for nucleotides and case-sensitive for amino acids,
    because that is what the C decode tables do: folding case is meaningful
    for bases and would be a silent widening for residues.
    """
    syms = symbols(alphabet)
    fold = alphabet == "nt"
    return all((c.upper() if fold else c) in syms for c in sequence)


def offenders(alphabet: str, sequence: str) -> list[str]:
    """The distinct characters of `sequence` that `alphabet` does not accept."""
    syms = symbols(alphabet)
    fold = alphabet == "nt"
    return sorted({c for c in sequence
                   if (c.upper() if fold else c) not in syms})
