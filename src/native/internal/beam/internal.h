/**
 * @file internal.h
 * @brief Shared layout and helpers for the BEAM implementation.
 *
 * Not part of the public API. Do not include from outside src/native/.
 *
 * ## Table layout, and why it is what it is
 *
 * Every table has the following LOGICAL indexing, packed
 * most-distant-context-first and matching the reference implementation exactly
 * so dense and sparse models can be compared cell by cell. The dense backend
 * is a flat `double` array holding every row. The sparse backend keeps the
 * same row index in a directory and stores only occupied rows, resolving an
 * absent one through its parent (see `beam_lp_row`). For a context of order
 * `k` at depth `d`, scoring residue `x`:
 *
 *     [ d ][ c_k ]...[ c_1 ][ x ]
 *
 * with `d` EXACT: the distance from the anchoring terminus, `i` forward and
 * `L-1-i` backward, and no clamp. Those two sum to `L-1`, which is why no
 * table is indexed by length.
 *
 * where `c_1` is the context residue NEAREST the scored one and `c_k` the
 * furthest. Context slots that fall outside the sequence hold the
 * alphabet's sentinel, `m->none_idx`.
 *
 * ONE table breaks that rule: the finalized C-anchored table is packed
 * nearest-first, `[ d ][ c_1 ]...[ c_k ][ x ]`. That is a performance
 * decision, it is confined to the finalized table (the counts and the file
 * format keep the common packing), and the reasoning is set out in full at
 * `beam_ctx_backward_finalized` below. Anything reading `lp[BEAM_BWD]` must
 * index it through one of the two helpers there, which are named apart for
 * exactly that reason.
 *
 * ## The two directions read opposite ways, and the packing follows
 *
 * For the forward (N-anchored) direction at position `p`, `c_j = s[p-j]`, so
 * the context is exactly the DAG state read oldest-first and a flat context
 * index is a suffix of the state index. For the backward (C-anchored)
 * direction, `c_j = s[p+j]`, so the digits come out of the state in the
 * REVERSE order. `beam_ctx_backward_counts_from_state` is the only place that
 * asymmetry lives; getting it backwards is the single most likely way to
 * produce a model that looks plausible and is wrong, so it is isolated and
 * unit-tested directly by `test_backward_context_from_state_matches_the_sequence`
 * in tests/c_unit/test_model.c — against hand-computed indices, against
 * the sequence form of the same context, and against the forward reading of
 * the same state, which it must NOT equal. Reading the state forwards leaves
 * the partition function normalising perfectly, so nothing else catches it.
 */
#ifndef BEAM_MODEL_INTERNAL_H
#define BEAM_MODEL_INTERNAL_H

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "beam/model.h"

#ifdef _OPENMP
#include <omp.h>
#endif

/*
 * The former contents of this header, split by responsibility. They are
 * included here in their original order so that every translation unit still
 * sees exactly what it saw before. Narrow one file at a time to the specific
 * headers it needs; this umbrella goes away when none remain.
 *
 * WHAT NARROWING BUYS, measured rather than assumed, because for most of this
 * header's life it bought nothing. Sixteen of nineteen internal headers
 * transitively included `model_state.h`, which publishes the whole
 * `struct BeamModel_` and pulls in the sparse-row and frozen-count
 * representations with it -- so swapping `internal.h` for `budget.h` took a
 * file from ten headers to five and left it seeing exactly as much. An audit
 * put it plainly: narrowing was a rename.
 *
 * That changed when `error_macros.h` stopped including `model_state.h` for one
 * macro none of its narrow consumers called, and `math_internal.h` stopped
 * including it for declarations that only take `const BeamModel *`. Today:
 *
 *     closure 0    alphabet.h  error_macros.h  math_internal.h
 *                  occupancy_internal.h  python_bindings.h  reduction.h
 *                  sparse_rows.h
 *     closure 1-2  budget.h  frozen_counts.h  query_internal.h
 *     pulls the model layout    indexing.h  live_frontier.h  model_build.h
 *                  memory.h  table_access.h  fb_sweep.h  mixing.h  dp.h
 *
 * Nine of twenty, down from sixteen of nineteen. So narrowing to any header in
 * the first two groups is now real encapsulation and worth doing; narrowing to
 * one in the third is still mostly a rename, and saying so is more useful than
 * promising a payoff the graph will not pay.
 */
#include "beam/model_state.h"
#include "beam/reduction.h"
#include "beam/sparse_rows.h"
#include "beam/frozen_counts.h"
#include "beam/live_frontier.h"
#include "beam/memory.h"
#include "beam/error_macros.h"
#include "beam/alphabet.h"
#include "beam/indexing.h"
#include "beam/table_access.h"
#include "beam/dp.h"
#include "beam/budget.h"

#endif /* BEAM_MODEL_INTERNAL_H */
