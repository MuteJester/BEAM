# Cross-model names and the relationship to FlashBack

## Cross-model summary names

A handful of accessors carry the spelling the graph models in this library use, so code that summarises several models side by side does not have to special-case BeamModel. Each is a real quantity, not a stub.

| accessor | quantity |
|---|---|
| `variant` | `"beam"` |
| `n_sequences` | total abundance ingested; an alias of `total_weight`, and the number a corpus-integrity check compares against a release's summed abundance column |
| `observed_lengths` | the lengths the model produces, ascending |
| `length_distribution` | `{length: pi(L)}` — the observed length distribution, **normalised**. BeamModel keeps the prior rather than the raw per-length totals, so a count would be a reconstruction. |
| `diversity_profile()` | now also carries `entropy_nats` alongside `D0`, `D1`, `D2` |
| `pgen_diagnostics()` | `is_proper`, `total_mass`, `length_prior_mass`, `max_abs_deviation`, `measure` |

`pgen_diagnostics` **measures** properness rather than asserting it: `sum_s P(s) == 1` is evaluated through the Mellin transform at `q=1`, which is a full sweep over the trellis, and `sum_L pi(L) == 1` through the length prior. A model that normalised only per length, or whose length prior had leaked, fails one of them. The reported `measure` is `"Pseq"`: BeamModel has no recombination model, so nothing in it is a Pgen.


## Relationship to FlashBack

BeamModel is a **separate model**, not a FlashBack variant. It keeps the geometric fact the FlashBack peel exposed on amino acids — that distance from each terminus is the informative coordinate — and drops the token fusion, which measurement showed costs about 900x the parameters for about 1% of the information. Nothing in the [FlashBackGraph](flashback-graph.md) or [FlashBackGrammar](flashback-grammar.md) implementation is shared or changed.

The C symbols are `BeamModel` / `beam_model_*` rather than `beam_*`, because `beam_v_prob` and friends already refer to *graph* edges.
