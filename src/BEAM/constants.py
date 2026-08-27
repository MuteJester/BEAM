"""Fixed dimensions of the model, mirrored from the C headers.
"""
from __future__ import annotations



from ._simulation_result import SimulationResult


BEAM_RESIDUES = 20

#: Longest sequence any BeamModel model can represent. Mirrors
#: ``BEAM_MAX_LENGTH`` in edge_model.h.
BEAM_MAX_LENGTH = 27

#: Largest supported context order for the wide expert. Mirrors
#: ``BEAM_MAX_ORDER``.
BEAM_MAX_ORDER = 4

#: Largest moment order the jet DP carries. Mirrors ``BEAM_MAX_MOMENT``.
BEAM_MAX_MOMENT = 16

#: The twenty amino acids the model accepts, in table order.
BEAM_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
