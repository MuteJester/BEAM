---
title: Errors
description: The exceptions BEAM raises, what each means, and which built-in exception each also is.
sidebar:
  order: 20
---

Errors from BEAM's computational core are raised as subclasses of `BEAM.BeamError` **and** of
the built-in exception they correspond to, so both `except BEAM.BeamError` and the matching
built-in (`except ValueError`, `except MemoryError`, …) catch them. A few argument checks made
in Python, such as `float()` of a [`BoundedValue`](/BEAM/reference/exact-occupancy/#boundedvalue),
raise the plain built-in (`ValueError`, `TypeError`). **Catching the built-in always works**; catch the
`BEAM` classes when you want to tell refusals apart.

```python
import BEAM

try:
    model.entropy()
except BEAM.BudgetExceeded as err:     # one specific refusal
    print(err)                         # the message says what was needed
```

| Exception | Means | Also a |
|---|---|---|
| `BeamError` | Base of BEAM's own refusals. | `Exception` |
| `BeamValueError` | An argument BEAM cannot use. | `ValueError` |
| `InvalidSequence` | A sequence BEAM cannot read: bad characters, empty, too long (for example in `explain`). | `ValueError` |
| `BeamTypeError` | An argument of the wrong type. | `TypeError` |
| `NotFinalized` | The model cannot be used this way, for example `save()` after `compact()`. | `RuntimeError` |
| `NoLivePaths` | Nothing satisfies the condition asked for, so the conditional does not exist. | `RuntimeError` |
| `ModelMismatch` | Two models cannot be combined: different alphabets, orders or shapes. | `RuntimeError` |
| `BudgetExceeded` | Refused because it would exceed the memory budget (4 GiB unless you set one). | `MemoryError` |
| `OutOfMemory` | The machine could not supply the memory. | `MemoryError` |
| `ModelFileError` | A model file could not be read or written. | `OSError` |
| `LikelihoodPieceError` | One piece of a [repertoire likelihood](/BEAM/reference/likelihood/) could not be computed; the total is unaffected. `SaddlepointDidNotConverge` is one kind. | `RuntimeError` |
| `InternalError` | A bug in BEAM; please report it. | `RuntimeError` |

Queries (`BeamQuery`) add `QueryError`, `ImpossibleCondition`, `UnsupportedQuery` and
`QueryTooLarge`; mixtures (`Mixture`) add `MixtureError`, `IncompatibleMembers`,
`UnsupportedOrder` and `MixtureTooLarge`. `ImpossibleCondition` is also a `NoLivePaths`;
`UnsupportedQuery` and `UnsupportedOrder` are `BeamValueError`s; `QueryTooLarge` and
`MixtureTooLarge` are `BudgetExceeded`; `IncompatibleMembers` is a `ModelMismatch`. All are
importable from `BEAM`.

## `BudgetExceeded` is not `OutOfMemory`

`BudgetExceeded` means BEAM declined because of the memory budget (4 GiB by default, set with
`BEAM.set_memory_budget`); raise the budget, ask at a lower order, or ask a smaller question.
`OutOfMemory` means the machine itself ran out, which a larger budget will not fix.

## Zero is an answer, not an error

`log_pseq` never raises: a sequence it cannot read (empty, longer than `max_length`, or with a
character outside the alphabet) scores `-inf`, a probability of zero. Every readable sequence,
including one never seen, has a non-zero probability, and every length from 1 to `max_length`
has some probability. What raises is a *conditional* on an impossible event, which does not
exist.

## Warnings, not errors

`BEAM.BeamPerformanceWarning` is a `UserWarning`, not a `BeamError`. It is not
a refusal: the build it precedes succeeds and returns the model you asked for.
It says the model will cost far more than its arguments suggest, which is worth
knowing before the wait rather than during it.

It is emitted for a build above 255 residues, because finalizing runs one
partition-function computation for every length from 1 to `max_length` — not
only the lengths your sequences have, since every length up to `max_length` has
some probability. The cost therefore grows with the square of `max_length`:
about 24 times the nucleotide default at 400 bases, and about 40 times at
512, counted in those computations rather than in seconds. It is also emitted
when `max_length="data"` stops at an alphabet's limit *and* some sequence is
longer than it, naming how many will be skipped.

Every route to such a model warns: the constructor, `from_dataframe`,
`BeamModel.builder`, `from_sequence_file`, `from_parquet` and `from_file` — a saved model stores
counts rather than tables, so loading one re-finalizes and pays the same cost.
Rebuilds from a model you already have, such as `without`, do not warn again.

Escalate it to an exception with

```python
warnings.filterwarnings("error", category=BEAM.BeamPerformanceWarning)
```

and silence it with
`warnings.simplefilter("ignore", BEAM.BeamPerformanceWarning)`. The
interpreter's `-W error::BEAM.BeamPerformanceWarning` does **not** work and is
discarded with "invalid module name": `-W` resolves categories before any
third-party package is importable. Under pytest, which re-parses `-W` after
collection, `-W error::BEAM.warnings.BeamPerformanceWarning` does work.
