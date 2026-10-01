# The probability of a repertoire

`repertoire_likelihood` answers `P(repertoire | model)` for an abundance-weighted model: its total, exactly, and the chain of conditional pieces that total factors into. `pair_likelihood` does the same for two repertoires under one model, with the sequences they share threaded through the chain.

```python
from BEAM import repertoire_likelihood, pair_likelihood

r = repertoire_likelihood(model, seqs, templates)
r.log_likelihood                     # exact
for p in r.pieces:
    p.name, p.log_prob, p.exact, p.observed, p.expected

atoms = union.pseq_analysis().atoms(8192)          # once per model
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

The total factors into four questions, each asked **given the N reads**:

```
P(counts | N) = P(n | N) x P(S | n, N) x P(F | S, N) x P(counts | S, F, N)
```

| Piece | Question | The summary it is about |
|---|---|---|
| `richness` | how many distinct sequences, given N reads | `n` |
| `identity` | which ones, given how many | the set `S` |
| `profile` | the abundance pattern, given which ones | `F`: how many were seen 1, 2, 3, 4, 5, 6-10, 11-100, over 100 times |
| `assignment` | which sequence has which count, given the pattern | the counts |

Each later summary determines the earlier ones: knowing the set `S` fixes `n`, and the counts fix `F`. So each conditional is a ratio, `P(later) / P(earlier)`, and nothing is counted twice. The four pieces multiply to `log_likelihood` exactly.

**Why given N.** The question is how surprising `n` distinct sequences are for a repertoire of N reads. Left free, the read total makes `n` look far more variable than it is wherever most reads are new sequences, which is where repertoires are. On the test suite's small model at N = 30 reads, `n` varies by 1.5 between repertoires drawn from the model, but by 4.8 if N is allowed to vary too. Against 20,000 repertoires drawn at fixed N, the richness piece's distribution of `n` is within sampling noise of the drawn one (total-variation distance 0.016 at N = 300). Leaving N free gave 0.15.

**How.** Let each sequence's count be independently Poisson with mean `lambda_s = N P(s)`. Then conditioning on the total is a ratio, `P(x | N) = P(x, N) / P(N)` with `P(N) = Poisson(N; N)`. Each piece is therefore built from joint probabilities of its summary *with* the read total:

```
richness   = P(n, N) / P(N)
identity   = P(S) P(N | S) / P(n, N)
profile    = P(F, N | S) / P(N | S)
assignment = P(counts | S) / P(F, N | S)
```

- `P(S)` is the observed sequences present and every other absent: `prod_S (1 - e^-lambda_s) x exp(-N (mass - sum_S P(s)))`. It is exact.
- `P(counts | S)` treats each seen sequence's count as a zero-truncated Poisson. It is exact.
- `P(n, N)` sums over every sequence the model can produce, about 1.4e35 for a donor model, grouped into the spectrum's atoms.
- `P(N | S)` and `P(F, N | S)` sum over the n seen sequences only.

The `poissonized_log_likelihood` is still reported: the probability of the counts with the total left free, which is `log_likelihood + log_prob_total_reads`.

## The clone-size spectrum, with no bins at all

```python
from BEAM import abundance_spectrum
sp = abundance_spectrum(model, seqs, templates)
sp.counts, sp.observed, sp.expected, sp.sd, sp.z
```

Given the set of sequences the repertoire was seen to hold, each count is an independent zero-truncated Poisson with mean `lambda_s = N P(s)`, so

```
E[F_k] = sum_s P(count_s = k)        Var[F_k] = sum_s p_sk (1 - p_sk)
```

are **exact** — plain sums over independent sequences, with no tilt, no saddlepoint and no binning. The whole spectrum of an Emerson donor (clone sizes 1 to 3,874 over 100,994 sequences) takes 0.02 s.

This is the quantity to compare between repertoires, and it is usually what you want rather than the `profile` piece. On Emerson donor Keck0001 under its own model:

| k | observed | E[F_k] | sd | z |
|---|---|---|---|---|
| 1 | 93,327 | 97,811.9 | 45.1 | −99.4 |
| 2 | 5,227 | 2,264.1 | 43.6 | +68.0 |
| 3 | 1,129 | 480.1 | 20.5 | +31.6 |
| 10 | 37 | 8.4 | 2.8 | +10.3 |

That is clonal expansion, read off per clone size.

It conditions on the set seen, **not** on the read total — that conditioning is the part that cannot be done exactly. `expected_reads` reports what the counts add up to under this reference, to be read beside `n_reads`.

**Why the chain still bins.** The joint `P(F = f)` for the full unbinned vector has generating function `prod_s (sum_k p_sk z_k)`, and extracting one coefficient is a permanent — #P-hard with distinct rates. A single profile *probability* therefore needs an approximation, and a saddlepoint needs few enough coordinates. Only the probability needs bins; the moments above never did.

## What is exact and what is not

**Exact:** the totals, `P(S)` and `P(counts | S)`.

**Approximate**, each labelled with its `method` and the size of its own `error` in nats:
- **The bins are chosen automatically** (`profile_edges="auto"`, the default) so the model expects at least five sequences in each. Fixed bins can be degenerate: "over 100 reads" is reached only by clones the model gives essentially no chance of, and a coordinate with no variance is one no tilt can move — the Hessian goes singular and the Newton oscillates between step limits. That lost 9.3% of one cohort's 846 donors outright and left others with a split worth nothing (reported error 1.3e7 nats). Automatic bins fixed every such case and cut the error fiftyfold on the rest (6.6 to 0.11 nats). They depend on the model and the repertoire, so a `profile` piece is comparable between repertoires only at the same `profile_edges`, which the result reports back.
- `P(n, N)`, `P(N | S)` and `P(F, N | S)` are sums of independent units, computed by a saddlepoint. Each unit is tilted so that the observed values become its mean, the normal density is taken there, and the tilt is undone exactly. The error is about `sum_j 1/(12 v_j)` over the tilted variances `v_j`. For the read total alone, which is skewed when the set is small, the second-order term is added and its size is reported. All three are held to exact convolution in the tests, on repertoires small enough to convolve. When the pattern fixes the total, because every non-empty bin is a single count, `P(F, N | S)` is computed without the joint. The same applies when the total is the least the pattern allows, every count at its bin's floor, or the most. Then every count is fixed, and `P(F, N | S)` becomes the probability that every count sits there. The joint saddlepoint is singular there, so this categorical sum is taken instead, with nearly fixed bins convolved exactly. On a test repertoire it lands 0.03 nats from exact, within its reported 0.04. Where a profile split's error still passes a nat, its `method` says the split is unreliable. Then read only profile plus assignment, which is `log P(counts | S) - log P(N | S)`: its one approximation is the small one in `P(N | S)`. An all-singleton repertoire's profile and assignment are exactly 0 and labelled exact.
- **Binning.** `P(n, N)` also uses the model's spectrum, binned. Its error is measured, not assumed. The piece is recomputed at half the bins, and a third of the difference is added to `error`, since the binning error falls with the square of the grid spacing. On an Emerson donor at 2048 bins the expected richness came out 85 sequences high (7.6e-4) and the piece 8 nats off. Hence the default of 8192 bins.
- Each joint appears in two neighbouring pieces with opposite signs. The approximations therefore move value between pieces and never change their sum.
- The `expected` abundance pattern is approximate. It uses each sequence's bin probabilities given that all the counts add up to N, holding the total by a normal approximation. Against exact convolution it is within 3% at forty sequences, and 9% in a nearly empty bin. Only the pattern's probability is held to exact convolution.

## What the model does not know

Reads of one clone are copies of one cell lineage, not independent draws, and one person's clone is large in both of their samples. The model has neither property. When they matter, the summaries land far from their expectations; the arithmetic is not at fault. Re-measured under the hierarchical prior on Emerson donor Keck0001, scored by its own model, the pattern holds. Its 128,553 reads have 100,994 distinct sequences against 111,376 expected given N, about 84 standard deviations short (`richness` -3,070 nats). It has 5,227 sequences read twice against 3,056 expected, and 227 read 11-100 times against 106. Read those pieces against a same-source control, not on their own.

## Cost

The spectrum (`model.pseq_analysis().atoms(bins)` and `.atoms(bins // 2)`) is the expensive step and depends only on the model, so pass them as `atoms=` and `atoms_coarse=` when scoring many repertoires against one reference. On a population model of 17M distinct sequences both resolutions together take about 2.5 minutes.

Given the spectrum, an Emerson-sized repertoire of about 100,000 distinct sequences takes **0.3 s**, almost all of it in two C kernels (`math/ztp_moments.c`): the tilted abundance-bin moments that every Newton step needs, and the expected pattern. They walk each sequence's counts once and return only the sums over sequences, so the cost no longer grows with the top profile edge — the default bins, up to 101 reads, cost what a much shorter set does. `pair_likelihood` still conditions its pieces on the Poisson total rather than on N.
