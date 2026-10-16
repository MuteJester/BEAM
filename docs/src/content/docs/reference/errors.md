---
title: Errors
description: The exceptions BEAM raises, what each means, and which built-in exception each also is.
sidebar:
  order: 20
---

Errors from BEAM's computational core are raised as subclasses of `BEAM.BeamError` **and** of
the built-in exception they correspond to, so `except ValueError` and `except BEAM.BeamError`
both catch them. Argument checks made in Python before reaching the core, such as
`model.explain()` on a sequence it cannot score or `float()` of a bounded value, raise the
plain built-in (`ValueError`, `TypeError`, `RuntimeError`). **Catching the built-in always
works**; catch the `BEAM` classes when you want to tell refusals apart.

```python
import BEAM

try:
    model.entropy()
except BEAM.BudgetExceeded as err:     # one specific refusal
    print(err)                         # the message says what was needed
```

| Exception | Means | Also a |
|---|---|---|
| `BeamError` | Base of every deliberate refusal. | `Exception` |
| `BeamValueError` | An argument BEAM cannot use. | `ValueError` |
| `InvalidSequence` | A sequence BEAM cannot read: bad characters, empty, too long. | `ValueError` |
| `BeamTypeError` | An argument of the wrong type. | `TypeError` |
| `NotFinalized` | The model cannot be used this way, for example `save()` after `compact()`. | `RuntimeError` |
| `NoLivePaths` | Nothing satisfies the condition asked for, so the conditional does not exist. | `RuntimeError` |
| `ModelMismatch` | Two models cannot be combined: different alphabets, orders or shapes. | `RuntimeError` |
| `BudgetExceeded` | Refused because it would exceed the memory budget you set. | `MemoryError` |
| `OutOfMemory` | The machine could not supply the memory. | `MemoryError` |
| `ModelFileError` | A model file could not be read or written. | `OSError` |
| `InternalError` | A bug in BEAM; please report it. | `RuntimeError` |

Queries (`BeamQuery`) add `QueryError`, `ImpossibleCondition`, `UnsupportedQuery` and
`QueryTooLarge`; mixtures (`Mixture`) add `MixtureError`, `IncompatibleMembers`,
`UnsupportedOrder` and `MixtureTooLarge`. Each also derives from the matching exception above.

## `BudgetExceeded` is not `OutOfMemory`

`BudgetExceeded` means BEAM declined because of the budget you set with
`BEAM.set_memory_budget`; raise it, or ask a smaller question. `OutOfMemory` means the machine
itself ran out, which a larger budget will not fix.

## Zero is an answer, not an error

`log_pseq` of a sequence the model cannot produce is `-inf`, a probability of zero.
`log_partition` and `log_length_prior` are `-inf` for a length the model does not produce.
What raises is a *conditional* on an impossible event, which does not exist.
