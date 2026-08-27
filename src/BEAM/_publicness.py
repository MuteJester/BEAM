"""Compatibility re-export.

PublicnessModel and its helpers now live in BEAM.publicness. This module
re-exports them so `from BEAM._publicness import ...` keeps working, including
SMALL_MU, which the test suite imports directly.
"""
from .publicness.constants import (
    MASS_TOL, SMALL_MU, TAIL_FLOOR, TAIL_SIGMA, _CHUNK_DOUBLES, _TINY_LEVELS,
)
from .publicness.cohort import _depth_groups, _next_power_of_two_above
from .publicness.model import PublicnessModel

__all__ = ["PublicnessModel", "SMALL_MU", "TAIL_SIGMA", "TAIL_FLOOR", "MASS_TOL"]
