---
title: Errors
description: The exceptions BEAM raises, what each means, and which built-in exception each also is.
sidebar:
  order: 20
---

Errors from BEAM's computational core are raised as subclasses of `BEAM.BeamError` **and** of
the built-in exception they correspond to, so both `except BEAM.BeamError` and the matching
built-in (`except ValueError`, `except MemoryError`, …) catch them. Some argument checks made in
Python, such as `model.explain()` on a sequence it cannot read or `float()` of a
[`BoundedValue`](/BEAM/reference/exact-occupancy/#boundedvalue), raise the plain built-in
(`ValueError`, `TypeError`, `RuntimeError`). **Catching the built-in always works**; catch the
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
| `InvalidSequence` | A sequence BEAM cannot read: bad characters, empty, too long. Scoring calls check first and raise a plain `ValueError`, so catch `ValueError` to cover every route. | `ValueError` |
| `BeamTypeError` | An argument of the wrong type. | `TypeError` |
| `NotFinalized` | The model cannot be used this way, for example `save()` after `compact()`. | `RuntimeError` |
| `NoLivePaths` | Nothing satisfies the condition asked for, so the conditional does not exist. | `RuntimeError` |
| `ModelMismatch` | Two models cannot be combined: different alphabets, orders or shapes. | `RuntimeError` |
| `BudgetExceeded` | Refused because it would exceed the memory budget (4 GiB unless you set one). | `MemoryError` |
| `OutOfMemory` | The machine could not supply the memory. | `MemoryError` |
| `ModelFileError` | A model file could not be read or written. | `OSError` |
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
