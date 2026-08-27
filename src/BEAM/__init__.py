"""BEAM: Bidirectional Expert-Anchored Model over CDR3 amino-acid sequences.

Every residue is scored twice -- once by an N-anchored expert indexed by its
distance from the front of the sequence, and once by a C-anchored expert
indexed by its distance from the back. The two readings are combined
log-linearly and normalised per length by an exact forward dynamic program
over a layered DAG. Nothing is sampled and nothing is pruned.
"""
from ._model import (
    BeamModel,
    Atoms,
    Builder,
    Calibration,
    Histogram,
    PseqAnalysis,
    effective_threads,
    memory_budget,
    threads,
    set_memory_budget,
    set_threads,
)
from ._publicness import PublicnessModel

__all__ = [
    "BeamModel", "Atoms", "Builder", "Calibration", "Histogram",
    "PseqAnalysis", "effective_threads", "memory_budget",
    "threads", "set_memory_budget", "set_threads",
    "PublicnessModel",
]
__version__ = "0.1.0"
