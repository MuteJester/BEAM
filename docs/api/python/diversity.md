# Diversity at a common sequencing depth

A Hill profile depends on how deeply a repertoire was sequenced. A donor sampled at 360,000 templates shows more distinct sequences — and a different profile — than the same donor at 330,000, and that difference rides straight into any comparison between two people. `depth_matched_hill_profiles` takes every repertoire in a set to one depth first, then profiles it.

```python
from BEAM import depth_matched_hill_profiles

r = depth_matched_hill_profiles(
    [seqs_a, seqs_b],                 # one list of distinct sequences per repertoire
    [templates_a, templates_b],       # integer template counts
    n_bootstraps=20, seed=20260916,
)
r.depth                   # the common depth: the smaller template total
r.profiles.shape          # (2, 20, 31): repertoire x resample x order
r.n_distinct_drawn        # (2, 20): distinct sequences each resample reached
band = r.resampling_band(5, 95)       # NOT a confidence interval -- see below
```

Model configuration passes through as keywords, exactly as to `BeamModel`: `alphabet="nt"`, `order=3`, `max_length=...`.

## What it computes, exactly

Put each sequence in a bag as many times as its abundance, draw `D` from the bag with replacement, fit a model to what was drawn, profile it, repeat:

```
D = min over repertoires of sum(abundances)          # templates, never distinct count
rng = numpy.random.default_rng(seed)                 # one generator
for each repertoire, in the order given:             # repertoire-major
    p = abundances / abundances.sum()
    for each of n_bootstraps replicates:             # replicate-minor
        draws  = rng.multinomial(D, p)
        hit    = numpy.flatnonzero(draws)
        model  = BeamModel([seqs[j] for j in hit], abundances=draws[hit], **config)
        profile = model.hill_profile(orders)
```

The deeper repertoires are thinned to the shallowest; the shallowest is never padded. Three properties are guaranteed and tested:

- **The draws are exactly the plain loop's**, given the same seed. They are numpy's, from one generator, in the order above, and every draw is made before any fit starts.
- **Every replicate is a fresh model** of exactly the sequences drawn, with exactly the drawn counts. Nothing — no count table, no support, no trellis — is shared between replicates. In the closed world the support is part of the answer, differs between replicates, and is `D0`.
- **The profiles do not depend on `n_workers`**, bit for bit: the draws are fixed before the threads start, each replicate is fitted on its own, and every result lands in its own slot.

`weighting="presence"` fits each resample with weight 1 per distinct sequence drawn instead of the drawn counts. The draw is abundance-proportional either way.

`seed=None` draws fresh entropy and records it in `r.seed`, so any result can be reproduced from its own record.

## The depth belongs to the call

!!! warning "Profiles are comparable only within one call"
    `D` is the smallest total abundance among the repertoires passed **in this call**. Adding or removing a repertoire can change `D`, and with it every other repertoire's profiles. Two results are comparable only if they were produced together — or at least share `depth` and configuration.

`D` counts templates, never distinct sequences. That is why abundances must be integers: a bag cannot hold 2.5 of a sequence. The rule also stops the commonest mistake here, since frequencies that sum to 1 would make `D` equal 1 and every replicate a model of a single sequence.

`k = 1` is allowed. `D` is then the repertoire's own total, and the result is a plain resample of it at its own depth.

## A resampling band, not a confidence interval

```python
band = r.resampling_band(lo=5, hi=95)
band.lower, band.median, band.upper     # (k, n_orders) each
```

!!! danger "This band says nothing about a new sample"
    Every replicate is a resample of **one** repertoire: the same sequences, drawn again in proportion to their abundance. The band therefore measures how much the profile moves when *what you already have* is reweighted — not what re-sequencing the same person would give. Measured at `q = 1` the two differ by an order of magnitude: the resampling band was about **3%** wide where held-out disjoint blocks of the same repertoire spread over about **50%**.

    Read it as a floor on the uncertainty, never as the uncertainty. It is named `resampling_band` rather than anything that could be read as an interval for a new sample, and neither result type defines `__float__`.

## What is refused, before anything is drawn

| input | why |
|---|---|
| a non-integer, zero, negative or non-finite abundance | `D` is a number of templates |
| a sequence listed twice in one repertoire | it would get two shares of the bag and be counted twice in `n_distinct_drawn`; merge and sum first |
| a sequence a fit would skip — too long, or outside the alphabet | every template of it a resample drew would silently vanish, leaving that resample short of `D` |
| `abundances=` or `sequences=` among the model keywords, or an unknown one | they are set per resample |

Nucleotide sequences are compared case-folded, as the model reads them, so `acgt` and `ACGT` are one duplicate.

## Threads

Each fit is itself parallel, and by default BEAM's OpenMP regions start one thread per core. Fitting replicates side by side on top of that multiplies the two: three unpinned processes on 48 cores once carried about 144 threads and ran about twice as slow per item. So the thread budget — the cap already set with `BEAM.set_threads`, otherwise what OpenMP would start — is **split**: `n_workers` replicates run at once, each with `budget // n_workers` threads, never more than the budget in total.

!!! note "The cap is set for the duration of the call"
    The split is applied through the process-wide [thread cap](runtime.md#parallelism) and restored afterwards, also on error. Other BEAM work in the same process sees the reduced cap meanwhile, and two of these calls must not overlap in one process. Results never depend on the cap.

Measured on a donor-scale pair — two synthetic repertoires of 180,000 distinct sequences each, 918,876 and 882,090 templates, amino-acid order 4, 20 resamples each, 16 threads:

| `n_workers` | threads per fit | wall time |
|---|---|---|
| 1 | 16 | 43.3 s |
| 2 | 8 | 29.8 s |
| 4 | 4 | 25.1 s |
| 8 | 2 | **23.4 s** |
| 16 | 1 | 23.4 s |

The profiles were bit-identical in every row. One fit's parallelism runs out long before sixteen threads, because its per-length sweeps are few and unequal, so replicates side by side are the better use of the cores. The default is therefore two threads per fit: `budget // 2` workers, at most eight. Eight of two is taken over sixteen of one because it holds half as many models in memory at once. The cap bounds that memory on a larger machine, where the best split has not been measured.

## Cost

A replicate is one fit and one profile. The profile is computed with [`hill_profile`](analytics.md#diversity), which walks each length's trellis once for all 31 orders; before it, the profile was 85% of every replicate. On the donor-scale pair above one replicate costs about 0.65 s to fit and 0.37 s to profile, where profiling with `hill_numbers` took 4.4 s.

!!! note "`hill_profile` agrees with `hill_numbers` to machine precision, NOT bit for bit"
    `hill_number` reaches `sum P**q` through a different tilted sweep that rounds differently. On the default grid the two agree to about `1e-14` relative, and near `q = 1` the relative difference grows as `1 / |1 - q|` on both routes. Profiles from this function will therefore not reproduce, to the last bit, profiles computed earlier by refitting and calling `hill_numbers` — they will agree to machine precision. See [Diversity](analytics.md#diversity).

## The result

`DepthMatchedProfiles` is frozen, its arrays are read-only, and it defines no `__float__`.

| field | shape | |
|---|---|---|
| `orders` | `(n_orders,)` | default `0, 0.1, ..., 3.0` |
| `depth` | | `D`, in templates |
| `weighting` | | `"abundance"` or `"presence"` |
| `n_bootstraps` | | resamples per repertoire |
| `seed` | | the seed the draws came from, recorded even when `None` was passed |
| `profiles` | `(k, n_bootstraps, n_orders)` | `D_q` of each resample |
| `n_distinct_drawn` | `(k, n_bootstraps)` | an output, not an input: a repertoire whose templates sit in few clones reaches fewer distinct sequences from the same draws |
| `n_templates`, `n_distinct` | `(k,)` | each repertoire's totals |
| `config` | | the model configuration, read-only |

`resampling_band(lo, hi)` returns a `ResamplingBand` with `lower`, `median` and `upper` of shape `(k, n_orders)`, and the percentiles, depth and replicate count it was computed from.
