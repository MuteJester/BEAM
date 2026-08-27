"""Normal density and distribution, for the saddlepoint inversion.
"""
from __future__ import annotations

import math


from .._simulation_result import SimulationResult


def _normal_pdf(z: float) -> float:
    return math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
