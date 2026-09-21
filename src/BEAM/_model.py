"""Compatibility re-export.

Everything here used to live in this module. It now lives in the packages
below, grouped by responsibility, and this module re-exports the same names so
that `from BEAM._model import ...` keeps working -- including the private
helpers the test suite reaches for directly.

New code should import from BEAM, or from the specific module.
"""
from .constants import (
    BEAM_ALPHABET, BEAM_MAX_LENGTH, BEAM_MAX_MOMENT, BEAM_MAX_ORDER,
    BEAM_RESIDUES,
)
from .runtime import (
    effective_threads, memory_budget, set_memory_budget, set_threads, threads,
)
from .io.readers import _column, _iter_delimited, _iter_parquet
from .math.saddlepoint import _normal_cdf, _normal_pdf
from .model.validation import _check_int, _is_scalar_arg
from .model.config import (
    _config_args, _config_args_from_dict,
)
from .model.facade import BeamModel
from .model.builder import Builder
from .analytics.histogram import Histogram
from .analytics.atoms import Atoms
from .analytics.pseq import PseqAnalysis
from .analytics.calibration import Calibration
from ._simulation_result import SimulationResult

__all__ = [
    "BeamModel", "Builder", "Histogram", "Atoms", "PseqAnalysis",
    "Calibration", "SimulationResult",
    "BEAM_ALPHABET", "BEAM_MAX_LENGTH", "BEAM_MAX_MOMENT", "BEAM_MAX_ORDER",
    "BEAM_RESIDUES",
    "effective_threads", "memory_budget", "set_memory_budget", "set_threads",
    "threads",
]
