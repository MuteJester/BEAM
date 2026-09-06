# Exact probabilistic queries

Every quantity here is computed by dynamic programming over the model's own trellis. Nothing is sampled, binned or fitted, and a question with no exact answer is refused rather than estimated.

```python
from BEAM import BeamModel, BeamQuery

q = BeamQuery(model)
c = q.condition(14).at(4, "G").contains("CASS")
q.probability(c)            # 0.0271...
q.marginals(c)              # (14, 20) array, every row a distribution
q.entropy(c)                # nats
```

## Conditions

A condition denotes a set of sequences of **one length**. It has two tiers, and the split is a cost ladder rather than a convenience.

**Masks are free.** `at`, `oneof`, `exclude` and `run` restrict what a position may hold. They delete edges from the trellis and add no states, so a condition built only from them costs exactly what an unconstrained query costs.

**Automata are not.** `contains` and `count` relate positions to each other, which no per-position restriction can express, so the query runs on a product graph costing a factor of the automaton's state count.

```python
c = q.condition(14)
c.at(3, "S")                     # position 3 is S
c.oneof(4, "FWY")                # position 4 is aromatic
c.exclude(5, "C")                # position 5 is not C
c.run(0, "CASS")                 # a prefix; run(L - n, ...) is a suffix
c.contains("QYF")                # the motif occurs somewhere
c.count("FWY", 2, 4)             # between 2 and 4 aromatic positions

c.is_free                        # nothing narrowed
c.needs_product                  # carries an automaton, so costs |Q| times more
c.pattern()                      # '___S._________' -- masks only
```

Every builder returns a **new** condition, so a partly built one can be shared and extended in several directions.

### Composition

```python
a & b        # both
a | b        # either
~a           # everything a does not select
```

`&` intersects masks, which is free. `|` and `~` cannot: the union of two per-position restrictions is not a per-position restriction, so both operands lift to automata first and the result costs a product graph.

## Probability

```python
q.probability(c)          # P(C | length)
q.log_probability(c)      # ln P(C | length); -inf for the impossible
q.log_partition(c)        # ln Z of the conditioned trellis
```

`log_probability` returns `-inf` rather than raising for a condition the model cannot produce. That is an **answer**: an event that cannot happen has probability zero. Conditioning on it is what has no answer, and every method below raises `ImpossibleCondition` instead.

## Distributions

```python
q.marginals(c)            # (length, n_residues); every row sums to 1
q.pair_joint(c, i, j)     # (n_residues, n_residues), summing to 1
```

A position the condition pins comes back one-hot, which is the correct conditional and not a placeholder. `pair_joint(c, i, i)` is diagonal, which is the correct joint of a position with itself.

## Information

```python
q.entropy(c)              # H(X | length, C), of the whole sequence
q.surprisal_variance(c)   # Var[-ln P] under the condition
q.position_entropy(c)     # H(X_i | C), one per position
q.mutual_information(c)   # (length, length), symmetric, non-negative
q.conditional_entropy(c, i, j)
q.total_correlation(c)    # sum_i H(X_i) - H(X)
```

### Three things are called "the entropy"

They are different numbers and the difference is not small.

| quantity | what it is |
|---|---|
| `q.entropy(c)` | `H(X \| L, C)` — conditional on the length, **excluding** the length prior |
| `length_marginals()["mean_surprisal"][L]` | `E[-ln P \| L]` on the **joint**, so larger by `-ln pi(L)` |
| `H(L) + sum_L pi(L) H(X \| L)` | the model's total entropy, which this engine does not produce |

At one measured length the first two differ by 1.54 nats — 33%. The variance is unaffected by a constant shift, so `surprisal_variance` matches the jet's exactly.

### `mutual_information` is within the event, not conditional on it

`q.mutual_information(c)` is the mutual information of the distribution **restricted to** `c`. It is not `I(X_i ; X_j | C)` in the information-theoretic sense, which averages over the values of `C`. On one test model an entry read 0.079 here against 0.157 for the averaged quantity. If you want the averaged form it is `p * MI(c) + (1 - p) * MI(~c)`, and `~c` is expressible.

`position_entropy` sums to **more** than `entropy` wherever positions share information; `total_correlation` is exactly that gap, and it sees dependencies of every order rather than only pairs.

## Expectations

```python
q.expect(table, c)                 # E[f], Var[f] for f(s) = sum_i table[i, x_i]
q.expect_count("FWY", c)           # E[# aromatic positions], and its variance
```

Only functionals that **decompose over positions** are exactly computable, and the table is all of them: indicators, residue counts and any weighted positional score are a table with different entries. An arbitrary function of a whole sequence is not here and will not be, because computing its expectation means enumerating the support or sampling it.

Every table entry must be **finite**. A residue that should not count wants a weight of zero, not the logarithm of a zero probability — `np.log(p)` gives `-inf` wherever `p` is zero, and a functional taking an infinite value on a live path has no variance and an infinite expectation.

## Insertions and deletions

```python
q.insertions("CASSF")     # every sequence one insertion away, scored
q.deletions("CASSF")
```

Each entry is a dict with `sequence`, `site`, `residue`, `multiplicity`, `log_odds`, `posterior`, `log_source` and `length_producible`, sorted by descending posterior.

`log_odds` is `ln P(s') - ln P(s)` on the **joint**, so the comparison across two lengths is well posed and the length prior does not cancel.

`posterior` is `P(s' | s' in this set)` — a distribution over **strings**, not over edit operations. Inserting `S` into the `SS` of `CASSSQYF` gives one string reachable four ways; it appears once, with `multiplicity = 4`. Weighted per edit instead — multiply by `multiplicity` and renormalise — that string is the likeliest neighbour, where per string it ranks ninth.

Both are right, under different questions. Per string asks what the model makes of a candidate set. Per edit asks what happens when a longer sequence is drawn and a uniformly chosen position removed, which is the story `PrismScorer` uses and why it sums repairs with multiplicity rather than deduplicating.

`site` is the **leftmost** gap producing that string — a canonical label, not a claim about where an edit occurred.

`length_producible` is false when the model generates no sequence of the neighbours' length at all. Every posterior is then zero for that one reason, which is a fact about the length prior rather than about any of these strings.

**None of this infers an edit.** The model has no alignment and no indel process. These are ratios between what it assigns to two strings that differ by one residue, describing where a repertoire tolerates insertions. For "given this observed sequence, where was it edited", use `PrismScorer`.

## Cost

```python
q.cost(c)                          # one query
q.cost(c, whole_matrix=True)       # a mutual-information matrix
```

Returns `states`, `edges` and `bytes`. `states` is an upper bound on what the sweep visits, not the live count — on a closed-world model it is a very loose one. `bytes` is the figure checked against the memory budget.

The ladder is worth seeing before you run into it: a mask-only condition is one sweep, an automaton multiplies that by its state count, and `mutual_information` is one sweep per `(position, residue)` pair — 281 rather than 1 on a length-14 amino-acid model.

## Refusals

| exception | means | base |
|---|---|---|
| `ImpossibleCondition` | the condition selects no sequence, so the conditional does not exist | `RuntimeError` |
| `UnsupportedQuery` | no exact answer, or malformed | `ValueError` |
| `QueryTooLarge` | exactly answerable, and over the memory budget | `MemoryError` |

Each keeps its built-in as a base class, so existing code catching `ValueError` around a query keeps working.

## Not yet available

- Cross-length questions: `E[length | C]`, `P(seq | C)`. Conditions are bound to one length at construction.
- Mutual information between a position and a **property** rather than another position.
- Exact KL divergence between two models.
- Constrained MAP — the most probable sequence satisfying a condition.
