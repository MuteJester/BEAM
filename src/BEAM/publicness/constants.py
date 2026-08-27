"""Numerical thresholds of the occupancy model, with their derivations."""
from __future__ import annotations
from typing import Any

from typing import Any



#: Half-width of the retained support, in standard deviations.
TAIL_SIGMA = 40.0

#: Absolute floor on the retained half-width, in occupancy levels. Covers the
#: degenerate ``mu ~ 0`` atoms, where sigma is ~ 0 but levels 0, 1, 2 ... still
#: hold real mass.
TAIL_FLOOR = 256.0

#: Mean occupancy below which the transform is bypassed for a closed form.
#:
#: The generating function of one atom is ``1 + mu*(z-1)``. Once ``mu`` falls
#: far below the double-precision epsilon of 1.0, that sum loses the whole
#: perturbation and the inverse transform returns quantisation residue rather
#: than a distribution. The residue is not even unbiased: rounding is an odd
#: nonlinearity applied to a cosine, so it lands on odd harmonics, and the
#: floor at odd occupancy levels runs about 1e12 times the floor at even ones.
#: Scaled by spectrum multiplicities of 1e17 and up, that becomes hundreds of
#: predicted sequences at occupancy levels a 1e-21 atom can never reach, in a
#: comb across alternate levels. The level-1 column degrades first and
#: separately: ``pmf[1]/mu`` falls to exactly 1/2 by ``mu ~ 3e-15``, because
#: the real part of the perturbation is annihilated while the imaginary part
#: survives intact.
#:
#: 1e-6 sits well above the onset of either failure, and both methods are
#: accurate there, so the crossover is continuous.
SMALL_MU = 1e-6

#: Occupancy levels retained by the closed form. ``mu**k / k!`` at ``SMALL_MU``
#: is below 1e-30 by k=5, so nothing above this carries representable mass.
_TINY_LEVELS = 5

#: A truncated PMF must still sum to one within this tolerance.
MASS_TOL = 1e-9

#: Doubles per PGF batch. One batch holds this many PMF entries and twice as
#: many generating-function entries, so roughly 48 MB at the default.
_CHUNK_DOUBLES = 1 << 21
