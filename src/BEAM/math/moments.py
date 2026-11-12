"""Length moments and the cumulant generating function of the surprisal.

The per-length derivative table, the CGF as the Mellin transform shifted by
one, and the jet of its first three terms at a tilt.

These are the implementations behind `BeamModel.length_moments`,
`BeamModel.cgf` and `BeamModel.cgf_jet`. They take the model as their first
argument rather than `self`; the methods remain on `BeamModel`, carry the
public documentation, and delegate here. Nothing in this module imports
`BeamModel` at runtime.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .. import _cbeam as _c
from ..constants import BEAM_MAX_MOMENT
from ..model.validation import _check_int

if TYPE_CHECKING:
    from typing import Any

    from ..model.facade import BeamModel


def length_moments(model: BeamModel, *, q: float = 1.0,
                   order: int = 4) -> dict[str, Any]:
    """Behind :meth:`BeamModel.length_moments`."""
    order = _check_int("order", order, 0, BEAM_MAX_MOMENT)
    b, rows, cols = _c.length_derivatives(model._g, float(q), order)
    raw = np.frombuffer(b, dtype=np.float64).reshape(rows, cols).copy()
    moments = raw.copy()
    share = raw[:, 0].copy()
    # Slot 0 carries the share instead of the constant 1 a raw jet would
    # put there, precisely so the mixture identity above is checkable from
    # the output alone. Restore the 1 in the moment view.
    moments[:, 0] = 1.0
    return {"length": np.arange(rows), "share": share, "moments": moments}


def cgf(model: BeamModel, t: float) -> float:
    """Behind :meth:`BeamModel.cgf`."""
    return model.log_mellin(1.0 + float(t))


def cgf_jet(model: BeamModel, t: float) -> dict[str, float]:
    """Behind :meth:`BeamModel.cgf_jet`."""
    lm, mean, var = _c.surprisal_tilt(model._g, 1.0 - float(t))
    return {"K": lm, "K1": mean, "K2": var}
