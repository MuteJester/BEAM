# Errors

Every exception BEAM raises deliberately descends from `BEAM.BeamError`, and every one of them also descends from the built-in it replaces. Catch whichever suits you:

```python
try:
    model.entropy()
except BEAM.BudgetExceeded as exc:      # a specific refusal
    ...
except BEAM.BeamError as exc:           # any refusal from BEAM
    ...
except ValueError as exc:               # still works, unchanged
    ...
```

The second base is what makes the hierarchy additive. Code written before it existed keeps working: `except ValueError` catches everything it caught before, `except MemoryError` still catches a budget refusal, and `except OSError` still catches a failed load.

## The hierarchy

| exception | means | also a |
|---|---|---|
| `BeamError` | base of every deliberate refusal | `Exception` |
| `BeamValueError` | an argument BEAM cannot use | `ValueError` |
| `InvalidSequence` | a sequence BEAM cannot read: bad characters, empty, too long | `ValueError` |
| `BeamTypeError` | an argument of the wrong type | `TypeError` |
| `NotFinalized` | the model is not finalized, or is compacted past this use | `RuntimeError` |
| `NoLivePaths` | nothing satisfies the question, so the conditional does not exist | `RuntimeError` |
| `ModelMismatch` | two models cannot be combined: different alphabets, orders or shapes | `RuntimeError` |
| `BudgetExceeded` | refused by `set_memory_budget`, not by the allocator | `MemoryError` |
| `OutOfMemory` | a genuine allocation failure: the machine said no | `MemoryError` |
| `ModelFileError` | a model file could not be read or written | `OSError` |
| `InternalError` | an invariant inside BEAM was violated — a bug in BEAM | `RuntimeError` |

The query engine adds three, each descending from both `QueryError` and its library equivalent, so `except BudgetExceeded` catches a query refused by the budget:

| exception | means | also a |
|---|---|---|
| `QueryError` | base of every refusal from the query engine | `BeamError` |
| `ImpossibleCondition` | the condition selects no sequence | `NoLivePaths`, `RuntimeError` |
| `UnsupportedQuery` | no exact answer, or malformed | `BeamValueError`, `ValueError` |
| `QueryTooLarge` | exactly answerable, and over the memory budget | `BudgetExceeded`, `MemoryError` |

## The distinction worth knowing

**`BudgetExceeded` is not `OutOfMemory`.** The first is BEAM refusing because you told it to; the remedy is yours — raise the budget, ask at a lower order, or ask a smaller question, and the message carries the figure needed and the budget in force. The second is the machine failing to supply memory, which raising the budget will not fix.

They shared an error code until recently, so code that wanted to tell them apart had to match on the words in the message. It no longer does.

## A refusal is not always an error

Some questions have an answer that happens to be extreme, and BEAM returns it rather than raising:

- `log_pseq` of a sequence the model cannot generate is `-inf`. That is a probability of zero, which is a fact about the sequence, not a failure.
- `log_clamp_probability` of an impossible clamp is `-inf`, for the same reason: `P(A)` is well defined when `A` cannot happen.
- `log_partition` and `log_length_prior` return `-inf` for a length the model does not generate.

What *is* raised is the conditional on an impossible event — `P(· | A)` does not exist — which is `NoLivePaths` or, through the query engine, `ImpossibleCondition`.

## Two gaps, stated rather than hidden

**Not every load failure is an `OSError`.** A file that is missing, unreadable, truncated, fails its checksum, or is in a format this build does not know raises `ModelFileError`. But a sound file that does not fit the memory budget raises `BudgetExceeded`, which is a `MemoryError` — the file is fine, your budget is not.

**A few allocation sites raise a bare `MemoryError` with an empty message.** They bypass BEAM's error context entirely, so `except BeamError` will not catch them and there is nothing to read. They are reachable by asking for an absurd length through the raw `_cbeam` layer. This is a known gap, pinned in `tests/python/test_error_contract.py` so it stays visible.

## For contributors

The C side is `include/beam/errors.h`: the `BeamError` enum, the thread-local message context, and the two calling protocols — most entry points return a code, six return a `double` sentinel and set the error out of band.

The code-to-class table lives in `src/native/bindings/python/common.c`. It supplies the **default** class for a code, and an entry point that means something more specific still raises that instead. One code does not imply one class: `BEAM_ERR_NO_LIVE_PATHS` is a `ValueError` from `positional_marginals`, an `ImpossibleCondition` from a query, and a plain `-inf` from `log_clamp_probability`. `tests/python/test_error_contract.py` pins the observed class per entry point, which is the thing that must not move.
