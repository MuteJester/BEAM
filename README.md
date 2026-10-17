<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/MuteJester/BEAM/main/docs/public/brand/beam-logo-dark.svg">
    <img alt="BEAM" src="https://raw.githubusercontent.com/MuteJester/BEAM/main/docs/public/brand/beam-logo-light.svg" width="360">
  </picture>
</p>

[![PyPI](https://img.shields.io/pypi/v/beam-model.svg)](https://pypi.org/project/beam-model/)
[![Python](https://img.shields.io/pypi/pyversions/beam-model.svg)](https://pypi.org/project/beam-model/)
[![Tests](https://github.com/MuteJester/BEAM/actions/workflows/tests.yml/badge.svg)](https://github.com/MuteJester/BEAM/actions/workflows/tests.yml)
[![License: GPL v3](https://img.shields.io/badge/license-GPL--3.0-blue.svg)](LICENSE)

**Every possible sequence, accounted for.**

BEAM is a probability model of biological sequences, built for T cell receptor
(TCRβ) CDR3s. It learns from the sequences alone, with their abundances if you
have them. Give it a sequence and it tells you how likely that sequence is; give
it a sequencing depth and it tells you how many distinct sequences you should
expect to see, how many two samples will share, and how many people in a cohort
will carry a given sequence. Each of these numbers is **summed over every
combinatorially possible sequence**, without listing or simulating them, so the
same input always gives the same answer.

BEAM comes with a ready-made reference model built from **38,538 human
repertoires**, and you can build models from your own data in one line.

> **Paper:** T. Konstantinovsky and G. Yaari. *A computable probability
> landscape for T cell receptor repertoire sampling and comparison.* (Submitted.)

## What can I do with it?

| Question | What BEAM gives you |
|---|---|
| How likely is this TCR? | The probability of any CDR3 sequence, normalised over all possible sequences |
| What do typical TCRs look like? | Realistic sequences drawn from the model |
| How deeply should I sequence? | The expected number of distinct sequences at any depth, and how many two samples share |
| How many people will share this TCR? | The expected number of repertoires in a cohort that contain it |
| How different are two repertoires, and why? | A distance that separates differences in sequence length from differences in local sequence patterns |

## Install

```bash
pip install beam-model
```

Ready-to-use packages exist for **Linux** (x86-64 and ARM), **macOS** (Apple
Silicon) and **Windows** (x86-64), for Python 3.11 to 3.15.

## Quick start

```python
import BEAM

model = BEAM.load_pretrained("foundation")    # downloads once (~90 MB), then cached

model.pseq("CASSLGQGYEQYF")                    # probability of one sequence
# 3.3e-05
```

The reference model uses about 1 GB of memory once loaded.

## Examples

### How likely is a TCR?

`pseq` returns the probability of a sequence under the model: the chance that
one TCR drawn at random has exactly this CDR3. It takes one sequence or a list.

```python
model.pseq(["CASSLGQGYEQYF", "CASSPGTGELFF", "CWWWWWWWWF"])
# array([3.3e-05, 2.9e-05, 6.9e-23])
```

Sequences the model has never seen still get a sensible probability: a
plausible new CDR3 gets a moderate one, an implausible one (like the last) a
vanishingly small one. Use `log_pseq_batch` for millions of sequences at once.

### What do typical TCRs look like?

```python
model.sample(5, seed=1)
# ['CASSPGTSSDTYEQYF', 'CASSLGTYSYNEQFF', 'CASRPGNSPLHF', ...]
```

Sequences are drawn in proportion to their probability.

### How deeply should I sequence?

If you sample *n* TCRs from a population, how many **different** sequences
will you see? And if two labs each sample *n*, how many sequences will they
have in common? BEAM computes both directly, without simulation.

```python
from BEAM.analytics.exact_occupancy import ExactOccupancy

occupancy = ExactOccupancy(model)

occupancy.richness(100_000).value               # distinct sequences in 100,000 TCRs
# 96771.4...
occupancy.richness(1_000_000).value             # ... and in a million
# 879776.1...
occupancy.sharing(100_000, 100_000).value       # shared by two samples of 100,000
# 3678.4...
```

Each result also reports a guaranteed bound on its error (`.bound`, here below
one part in a trillion). The first query takes a minute or two while BEAM
computes the integrals it needs; every later query, at any depth, is instant.

### How many people will share a TCR?

A sequence is *public* when many people carry it. Describe a cohort by how
deeply each person's repertoire was sampled, and BEAM tells you how many of
them are expected to contain a sequence.

```python
from BEAM import PublicnessModel

cohort = PublicnessModel([50_000] * 100)        # 100 people, 50,000 sequences each

mean, variance = cohort.moments(model.pseq("CASSLGQGYEQYF"))
mean                                            # expected number of people
# 80.8
```

`cohort.pmf(p)` gives the full distribution: the probability that exactly 0,
1, 2, ... people share the sequence.

### How different are two repertoires, and why?

Build a model from each person's repertoire and compare them. The distance
splits exactly into a part due to **different sequence lengths** and a part
due to **different local sequence patterns**, and it names the patterns that
differ most.

```python
import pandas as pd
from BEAM import BeamModel
from BEAM.analytics.distance import compare

def repertoire_model(path):
    table = pd.read_csv(path, sep="\t")             # an AIRR-format table
    return BeamModel.from_dataframe(table, sequence_column="junction_aa",
                                    abundance_column="duplicate_count")

a = repertoire_model("donor_A.tsv")
b = repertoire_model("donor_B.tsv")

result = compare(a, b)
result.distance                                     # 0 = identical
result.length_squared_distance / result.squared_distance   # share due to lengths
result.patterns[:5]                                 # the patterns that differ most
```

Comparing two repertoires takes about a minute.

## Using your own data

The reference model was built from CDR3 amino-acid sequences that **include**
the conserved cysteine at the start and the phenylalanine or tryptophan at the
end, for example `CASSLGQGYEQYF`; use the same convention when you score
sequences against it or build models to compare with it. In AIRR-format files
this is the `junction_aa` column (AIRR's `cdr3_aa` column leaves those two
residues out).

```python
from BEAM import BeamModel

model = BeamModel(sequences, abundances=counts)     # from Python lists
model = BeamModel.from_sequence_file("cdr3.tsv")    # one "SEQUENCE<TAB>COUNT" per line
model.save("my_repertoire")                         # -> my_repertoire.beam
model = BeamModel.from_file("my_repertoire.beam")
```

Abundances are optional; without them every sequence counts once. Nucleotide
sequences work too, with `alphabet="nt"`.

## Why no simulation?

Most questions about a repertoire are sums over every possible sequence, and
there are far too many sequences to list. The usual workaround is simulation:
draw many sequences and count. BEAM is built so that these sums can be
computed directly, by dynamic programming over the whole sequence space. Its
probabilities are normalised over all sequences, and quantities such as
expected richness, overlap and publicness are evaluated numerically, with
stated error bounds where a sum is approximated. The same input always gives
the same answer, and small differences between groups are not lost in
simulation noise.

## Documentation

The API reference is in [`docs/api/python/`](docs/api/python/):
[the model](docs/api/python/model.md),
[scoring and analytics](docs/api/python/analytics.md),
[publicness](docs/api/python/publicness.md),
[the probability of a whole repertoire](docs/api/python/likelihood.md),
[mixtures of repertoires](docs/api/python/mixture.md),
[diversity](docs/api/python/diversity.md) and
[queries](docs/api/python/query.md).

Questions and ideas are welcome in
[Discussions](https://github.com/MuteJester/BEAM/discussions); bugs in
[Issues](https://github.com/MuteJester/BEAM/issues).

## Citing BEAM

If BEAM is useful in your work, please cite the paper above. GitHub's **"Cite
this repository"** button (from [`CITATION.cff`](CITATION.cff)) gives the
reference in BibTeX and APA.

## License

BEAM is free software under the [GNU General Public License v3.0 or
later](LICENSE). Contributions are welcome: see
[`CONTRIBUTING.md`](CONTRIBUTING.md).
