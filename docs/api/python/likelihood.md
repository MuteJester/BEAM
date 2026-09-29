# The probability of a repertoire

`repertoire_likelihood` answers `P(repertoire | model)` for an abundance-weighted model: its total, exactly, and the chain of conditional pieces that total factors into. `pair_likelihood` does the same for two repertoires under one model, with the sequences they share threaded through the chain.

```python
from BEAM import repertoire_likelihood, pair_likelihood

r = repertoire_likelihood(model, seqs, templates)
r.log_likelihood                     # exact
for p in r.pieces:
    p.name, p.log_prob, p.exact, p.observed, p.expected

atoms = union.pseq_analysis().atoms(2048)          # once per model
pair = pair_likelihood(union, seqs_a, templates_a, seqs_b, templates_b, atoms=atoms)
pair.piece("sharing").observed, pair.piece("sharing").expected
```

## The total

The model is abundance-weighted, so `P(s)` is the probability of one read. N reads are N independent draws, and

```
log P(R | M) = log N! - sum_s log c_s! + sum_s c_s log P(s)
```

- `N` is the total read (template) count, `c_s` the reads of sequence `s`.
- `sum_s c_s log P(s)` is the chance of one particular order of the N reads.
- `log N! - sum_s log c_s!` counts the orders that give the same counts. It does not depend on the model, so it cancels whenever two models score the same repertoire.

That is `log_likelihood`. It is `-inf` only when the model cannot represent one of the sequences at all: a character outside its alphabet or a length past `max_length`. Under the hierarchical prior every other sequence is possible, so one donor's model can score another donor's repertoire directly.

## The pieces

Let the total be free: every sequence independently Poisson with mean `lambda_s = N P(s)`. That changes the total by exactly `log Poisson(N; N)` (`log_prob_total_reads`), and the Poisson version factors as

```
P(counts) = P(n) x P(S | n) x P(F | S) x P(counts | S, F)
```

| Piece | Question | The summary it is about |
|---|---|---|
| `richness` | how many distinct sequences | `n` |
| `identity` | which ones, given how many | the set `S` |
| `profile` | the abundance pattern, given which ones | `F`: how many were seen 1, 2, 3, 4, 5, 6-10, 11-100, over 100 times |
| `assignment` | which sequence has which count, given the pattern | the counts |

Each later summary determines the earlier ones: knowing the set `S` fixes `n`, and the counts fix `F`. So each conditional is a ratio, `P(later) / P(earlier)`, and nothing is counted twice. The pieces sum to `poissonized_log_likelihood` to rounding.

For two repertoires under one model, `sharing` is inserted after `richness`: `P(k shared | n1, n2)`, with `k` the number of distinct sequences in both.

## What is exact and what is not

**Exact:**
- all three totals;
- `P(S)`, the observed sequences present and every other absent: `prod_S (1 - e^-lambda_s) x exp(-N (mass - sum_S P(s)))`;
- `P(counts | S)`, each seen sequence a zero-truncated Poisson;
- `shared_counts_log_prob` and `private_counts_log_prob`;
- the `expected` of each summary, up to the spectrum binning (next paragraph).

**Approximate**, each labelled `exact=False` with its `method`:
- `richness`, `sharing` and `profile` are the probability that independent units fall into given category totals. They are computed by a saddlepoint, and each piece reports the size of its own error in `error`: `sum_j 1/(12 v_j)` nats over the categories its normal term covered, `v_j` their variance under the saddlepoint's tilt. Against exact convolution over some six thousand totals the actual error never exceeded 0.97 of it. A category with tilted variance below four (the 150-read clone in a bin the model gives 1e-100) is convolved exactly instead, so it contributes no error.
- `richness` and `sharing` also sum over the model's binned spectrum, and that error is NOT in `error`. The relative error of their expectations is about `spacing^2 / 8` (4e-6 at the default 2048 bins, falling at least sixteenfold per fourfold increase in bins), but far in a tail it is amplified: a repertoire ten standard deviations below its expected richness saw 0.009 nats at 2048 bins and 0.0005 at 8192. Ask for more bins when a far-tail value matters.
- `identity` and `assignment` are exact quantities divided by those approximations. They therefore carry the same error with the opposite sign, and their sums with the pieces before them are exact.

## What the model does not know

Reads of one clone are copies of one cell lineage, not independent draws, and one person's clone is large in both of their samples. The model has neither property. When they matter, the summaries land far from their expectations; the arithmetic is not at fault. On Emerson donors, measured under the previous closed-world model, a repertoire's own abundance-weighted model gave its largest clones more probability than their reads and its singletons less, so `richness` and `profile` missed even under the repertoire's own model; that has not been re-measured under the hierarchical prior. Read those pieces against a same-source control, not on their own.

## Cost

The spectrum (`model.pseq_analysis().atoms(bins)`) is the expensive step and depends only on the model: pass it as `atoms=` when scoring many repertoires against one reference. Given the spectrum, a pair of Emerson repertoires of about 100,000 distinct sequences each takes about 2.5 s. The profile piece costs one pass per read count up to the last edge.
