# Insights: plottable views of a model

Derived quantities you can act on, built from the exact machinery already in the library. Nothing here computes anything new about a model, and nothing here draws: it returns arrays, and you plot them in whatever you like.

```python
s = model.insights.logos(15, pseq_below=1e-9, bins=4096)
s[0].as_dict()              # {residue: heights}, ready for logomaker
s[0].consensus_sequence     # 'CASSLGGGGGYEQYF'
s[0].fraction               # 0.30 — how much of length 15 this band is
```

## Sequence logos

For each position of a given length, the residue distribution and the information content of the stack:

```
information[i] = log(reference_residues) - H[i]
height[i][a]   = probability[i][a] * information[i]
```

In bits that is the Schneider–Stephens logo every plotting package expects. Both units come back, named — `height_bits` and `height_nats`, and the same for `entropy` and `information`.

**No small-sample correction is applied and none is wanted.** The classic `e_n = (R-1)/(2 ln2 · n)` corrects a logo built from `n` observed sequences. This one is analytic over the whole support, so `n` is unbounded and the correction is zero.

**`information` is a relative entropy against a uniform background.** `log R − H` is exactly `D_KL(p || uniform)`. That is a choice, and naming it is the point: a reader who does not know which background was used cannot tell that one was.

## Conditioning on Pseq

`pseq_above=t` keeps sequences with `Pseq > t`; `pseq_below=t` keeps `Pseq <= t`. They differ in strictness so that together they partition, and the strictness sits below the resolution of the method anyway — a sequence within a bin of the edge is split across it.

```python
common = model.insights.logo(15, pseq_above=1e-8, bins=4096)
rare   = model.insights.logo(15, pseq_below=1e-8, bins=4096)
common.fraction + rare.fraction     # 1.0
```

Surprisal is the **native** axis, so if you are already thinking in nats there is nothing to convert:

```python
rare = model.insights.logo(15, surprisal_above=20.7)     # nats
rare = model.insights.logo(15, pseq_below=1e-9)          # the same request
```

Note that **"above" means opposite things on the two axes.** Rarer is higher surprisal and lower probability, so `surprisal_above` and `pseq_below` select the same sequences, as do `surprisal_below` and `pseq_above`. That is why each keyword is named for its own axis rather than sharing one word, and why giving one of each is refused rather than resolved.

For full control, `edges=` takes ascending surprisal in nats, and `n` edges give `n − 1` logos. `±inf` is allowed at either end and is resolved against the grid:

```python
model.insights.logos(15, edges=[-inf, 16.1, 18.4, 20.7, inf])   # four bands
```

Two things to know about the threshold:

- It is on the **joint** `Pseq`, the quantity `log_pseq` returns, which carries the length prior. So one fixed `t` selects a different fraction at every length, and at a rare length it may select nothing.
- Surprisal is `−ln p`, so it **ascends as probability falls**. The keywords handle the flip; `edges=` does not.

## Exact, or on a grid

The unconditional logo of the generated measure is **exact** — no grid is involved at all, and it costs one forward-backward sweep. Any Pseq condition needs the surprisal grid.

```python
s = model.insights.logos(15)
s.exact                     # True
s[0].resolution_error       # None — there was no grid to be wrong about
```

`exact` describes the answer, not which function produced it: a single band covering the whole grid is exact too, because there is no interior edge for a sequence to be misclassified across.

When a grid *is* used, two different faults can occur and neither implies the other:

| symptom | fault | fix |
|---|---|---|
| `logo.resolution_error` large | grid too **coarse** — band edges cut through mass the grid cannot see | more `bins`, or wider bands |
| `set.grid_shortfall` above zero | grid too **short** — mass ran off the top | more `bins`, or a larger `spacing` |

Pass `max_resolution_error=` to have a too-coarse grid refused rather than returned with a caveat you have to remember to read.

## How many bins

Linear splitting spreads one sequence over `length + 1` bins, so what governs the answer is that kernel against the width of a band:

```
kernel  = (length + 1) * spacing
spacing = accumulator_max * 1.02 / (bins - max_length - 9)

bins >~ 5 * (length + 1) * accumulator_max / band_width
```

Measured on the 927M-sequence foundation model (order 4, built under the previous closed-world rule) at its modal length 15, with quartile bands about 2.4 nats wide:

| bins | spacing | `resolution_error` | scratch | time |
|---|---|---|---|---|
| 1024 | 0.122 | 3.4% | 16 GB | 7.7 s |
| 2048 | 0.060 | 2.2% | 31 GB | 10 s |
| **4096** | 0.030 | **1.1%** | 61 GB | 19 s |
| 8192 | 0.015 | 0.44% | 119 GB | 37 s |

Halving the spacing halves the error, which is what says it is the grid and not a defect. Band fractions are stable to a tenth of a point from 4096 on. **The default of 512 is not enough for that model**: its kernel there is 4.04 nats against bands of 2.4, so the smoothing is wider than the band it is meant to resolve.

Neither model size nor strictness decides this. Two things do: how wide a surprisal range the grid must span, and whether there is any mass within a kernel of the edge. A small model whose spectrum is a few dozen spikes has edges that land in the gaps and pays nothing for a coarse grid.

## Cost, before you pay it

```python
model.insights.cost(15, pseq_below=1e-9, bins=4096)
# {'states': ..., 'edges': ..., 'bytes': 61_000_000_000}
```

`bytes` is the figure that will be checked against `BEAM.memory_budget()`, so compare and warn rather than catch a refusal. For a banded sweep it is exact: a budget of `bytes` runs it and one byte less refuses it. That is because it is measured, not bounded. The backward rows are stored at their own widths, and the estimate runs the sweep's integer pass over the edges to size them.

## Nucleotides

BEAM has **five** nucleotide residues, because `N` is emitted and learned rather than treated as a wildcard. So the default ceiling is `log2(5) = 2.32` bits where every published DNA logo tops out at 2. Ask for the conventional axis:

```python
model.insights.logos(30, reference_residues=4)
```

`information` may then be negative at a position that spells `N` with real probability. That is information about the model, and it is not clamped.

## What is refused

A logo of nothing would come back as **maximum information content at every position** — a blank plot beside a summary claiming perfect conservation. So an empty band is refused, not returned. So is a band outside the surprisal a length can attain, and that one is refused *before* the sweep allocates anything.

Also refused: edges that do not ascend, a NaN edge, a negative edge (almost always `ln(p)` where `−ln(p)` was meant), a reference wider than the alphabet, and `length=0` — logos are per length, because position 5 of a length-12 sequence and position 5 of a length-18 one are different distances from the C terminus.
