# PublicnessModel

Expected sharing across a cohort of repertoires: given per-sequence
probabilities and the depths of the repertoires they might appear in, how
many sequences are expected at each occupancy level.

BEAM's spectrum atoms are the intended input. `PseqAnalysis.atoms()` returns
the `(probabilities, multiplicities)` pair this consumes with no reshaping.

```python
from BEAM import BeamModel, PublicnessModel

p, w = model.pseq_analysis().atoms(2048).as_arrays()
counts = PublicnessModel(depths).expected_counts(p, w)
```

## Methods

### `bin_mass(p: 'float', edges: 'Any') -> 'np.ndarray'`

Probability the occupancy count falls in each half-open bin.

### `depth_groups() -> 'tuple[np.ndarray, np.ndarray]'`

Representative depths and their multiplicities.

### `depth_spread`

*property.* Widest within-group ratio of depths. ``1.0`` means exact grouping.

### `expected_counts(probabilities: 'Any', multiplicities: 'Any', *, levels: 'Any' = None) -> 'dict[str, Any]'`

Predicted sequences per publicness bin, over a whole spectrum.

``probabilities`` and ``multiplicities`` are a p-sequence spectrum:
one probability atom and the number of distinct sequences carrying it.
Atoms of zero multiplicity are dropped before any work is done, which
matters because a deterministic surprisal grid is mostly empty.

``levels`` selects the publicness binning:

* ``None`` (default) gives one bin per occupancy level, 0..R.
* an ``int`` gives that many equal-width bins spanning 0..R+1, merged
  where rounding would produce empty ones.
* a sequence gives explicit non-decreasing integer bin edges, read as
  half-open ranges ``[edges[i], edges[i+1])``.

### `expected_counts_by_group(probabilities: 'Any', multiplicities: 'Any', groups: 'Any', *, n_groups: 'int | None' = None, levels: 'Any' = None) -> 'dict[str, Any]'`

Predict publicness counts for several atom groups together.

Each probability atom contributes only to the output row named by its
non-negative integer ``groups`` value. The probability generating
functions and FFT inversion are batched across all rows, while the
native accumulator applies the same moment-derived truncation, clip
order, and retained-mass test as :meth:`expected_counts` before adding
each atom to its row.

This is mathematically equivalent to selecting each group and calling
:meth:`expected_counts` separately. It is intended for joint surfaces
such as P-sequence probability bin by repertoire publicness.

Args:
    probabilities: Sequence probability for every spectrum atom.
    multiplicities: Number of sequences represented by every atom.
    groups: Non-negative integer output row for every atom.
    n_groups: Number of output rows. Defaults to one above the largest
        group index; larger values retain explicit empty rows.
    levels: Publicness binning accepted by :meth:`expected_counts`.

Returns:
    Diagnostics and an ``expected_counts`` float64 matrix with shape
    ``(n_groups, number_of_publicness_bins)``.

### `k_fft`

*property.* Number of roots of unity used to invert the generating function.

### `moments(p: 'Any') -> 'tuple[Any, Any]'`

Closed-form mean and variance of the occupancy count.

Neither touches the PMF, which is exactly why they can be trusted to
say which part of the inverted PMF is signal.

### `n_depth_groups`

*property.* Number of factors in the generating-function product.

### `n_repertoires`

*property.* Number of repertoires in the cohort.

### `pmf(p: 'float') -> 'np.ndarray'`

Occupancy PMF of one sequence probability, over levels 0..R.

Truncated, clipped and mass-checked exactly as the aggregate is, so
this is the single-atom view of what :meth:`expected_counts` sums.
