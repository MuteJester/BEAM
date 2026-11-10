"""The warnings BEAM emits.

This module is named ``warnings`` and does not shadow the standard library:
Python 3 resolves ``import warnings`` inside the package absolutely, and only
``from . import warnings`` reaches this file. It mirrors :mod:`BEAM.errors`.
"""
from __future__ import annotations


class BeamPerformanceWarning(UserWarning):
    """A build that will succeed and will cost far more than the caller is
    likely to expect.

    It subclasses :class:`UserWarning`, so it is shown by default. Escalate it
    with ``warnings.filterwarnings("error", category=BEAM.BeamPerformanceWarning)``
    and silence it with
    ``warnings.simplefilter("ignore", BEAM.BeamPerformanceWarning)``.

    Not with ``-W error::BEAM.BeamPerformanceWarning``: the interpreter
    resolves ``-W`` categories while :mod:`warnings` initialises, before any
    third-party package is importable, and rejects the option outright
    ("invalid module name: 'BEAM'"). The same is true of ``PYTHONWARNINGS``
    and of every other package's categories. pytest re-parses ``-W`` after
    collection, so ``-W error::BEAM.warnings.BeamPerformanceWarning`` does
    work there, and this project's suite relies on it.
    """
