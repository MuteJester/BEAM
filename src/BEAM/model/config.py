"""Configuration assembly and the smoothing defaults.
"""
from __future__ import annotations

from typing import Any


from ..constants import BEAM_MAX_LENGTH, BEAM_MAX_ORDER, BEAM_MAX_MOMENT
from .validation import _check_int

from .._simulation_result import SimulationResult


def _config_args(*, order, depth_cap, max_length, coefficients, kappa,
                 kappa_wide, kappa_len, weights, strict=True,
                 sparse=None) -> tuple:
    """Validate a build configuration and flatten it for the C entry points.

    Shared by :class:`BeamModel` and :class:`Builder` so a streamed build and a
    one-shot build cannot be validated differently, which would make the
    "byte-identical" claim on :meth:`Builder.build` a coincidence.
    """
    if abs(sum(coefficients) - 1.0) > 1e-9:
        raise ValueError(
            f"coefficients must sum to 1, got {sum(coefficients)!r}. The "
            "sum is a cross-length calibration knob, not a free "
            "parameter; moving it off 1 degrades global ranking while "
            "leaving within-length ranking untouched.")
    order = _check_int("order", order, 2, BEAM_MAX_ORDER)
    # 2 is a floor, not an oversight: `depth_cap` must be at least 1 and
    # strictly below `max_length`, so a one-residue model has nowhere to
    # put its wide expert's depth resolution.
    max_length = _check_int("max_length", max_length, 2, BEAM_MAX_LENGTH)
    depth_cap = _check_int("depth_cap", depth_cap, 1, BEAM_MAX_LENGTH - 1)
    # A cap at or beyond the longest sequence resolves every depth exactly,
    # so the C layer clamps it; mirroring that here keeps `config` honest
    # about what was actually built.
    depth_cap = min(depth_cap, max_length - 1)
    coefficients = tuple(float(c) for c in coefficients)
    kappa = tuple(float(k) for k in kappa)
    kappa_wide = float(kappa_wide)
    kappa_len = float(kappa_len)
    weights = tuple(float(w) for w in weights)
    strict = bool(strict)

    zero_backoff = all(k == 0.0 for k in kappa) and (
        kappa_wide == 0.0 and kappa_len == 0.0)
    if sparse is None:
        sparse = strict and zero_backoff
    else:
        sparse = bool(sparse)
        if sparse and not (strict and zero_backoff):
            raise ValueError(
                "sparse=True requires strict=True and all five backoff "
                "strengths (kappa, kappa_wide, and kappa_len) to be zero")

    return (order, depth_cap, max_length, coefficients, kappa,
            kappa_wide, kappa_len, weights, strict, sparse)


def _smoothing_defaults(strict, kappa, kappa_wide, kappa_len):
    """Resolve the backoff strengths, whose default depends on `strict`.

    Open-world (the default) wants strong backoff: 20/20/20 and 200. Strict
    mode wants none, because backoff from a shorter context is exactly the
    thing that gives an unobserved transition mass and stops D0 from meaning
    anything. Passing kappa explicitly always wins, which is what makes the
    partial regime -- strict with backoff still on -- reachable for anyone
    who wants it.
    """
    if kappa is None:
        kappa = (0.0, 0.0, 0.0) if strict else (20.0, 20.0, 20.0)
    if kappa_wide is None:
        kappa_wide = 0.0 if strict else 20.0
    if kappa_len is None:
        kappa_len = 0.0 if strict else 200.0
    return kappa, kappa_wide, kappa_len


def _config_args_from_dict(cfg: dict[str, Any]) -> tuple:
    """The same tuple, rebuilt from what :attr:`BeamModel.config` reports."""
    return _config_args(order=cfg["order"], depth_cap=cfg["depth_cap"],
                        max_length=cfg["max_length"],
                        coefficients=cfg["coefficients"], kappa=cfg["kappa"],
                        kappa_wide=cfg["kappa_wide"],
                        kappa_len=cfg["kappa_len"], weights=cfg["weights"],
                        strict=cfg.get("strict", True),
                        sparse=cfg.get("sparse", False))
