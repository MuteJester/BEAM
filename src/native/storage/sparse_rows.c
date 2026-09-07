/**
 * @file sparse.c
 * @brief Dense-directory, sparse-payload storage for strict BEAM tables.
 */

#include "beam/internal.h"

/* A small first allocation avoids penalising one-row partitions while keeping
 * the normal observe path from allocating for every new row. */
#define BEAM_SPARSE_INITIAL_CAPACITY 8u

static bool sparse_array_bytes(size_t count, size_t element_size,
                               size_t *out) {
    if (element_size != 0u && count > SIZE_MAX / element_size) return false;
    *out = count * element_size;
    return true;
}

static bool sparse_value_bytes(uint32_t rows, uint32_t stride, size_t *out) {
    const size_t row_bytes = (size_t)stride * sizeof(double);
    return sparse_array_bytes((size_t)rows, row_bytes, out);
}

/**
 * Grow one independently-owned partition without exposing a half-grown pair
 * of arrays.  In particular, realloc cannot be used here: values may move
 * successfully and the matching row-id allocation may then fail.  Allocate
 * both replacements, copy the occupied prefix, and only then commit them.
 */
static bool sparse_grow_partition(BeamSparseRows *t, uint32_t part,
                                  uint32_t needed) {
    if (!t || part >= t->n_parts || !t->values || !t->size ||
        !t->capacity || needed == 0u || needed > t->n_dense_rows)
        return false;

    const uint32_t old_capacity = t->capacity[part];
    const uint32_t occupied = t->size[part];
    if (occupied > old_capacity) return false;
    if (old_capacity >= needed) return true;
    if (occupied != 0u && !t->values[part]) return false;
    if (t->mutable_rows &&
        (!t->row_ids || (occupied != 0u && !t->row_ids[part])))
        return false;

    uint32_t new_capacity = old_capacity;
    if (new_capacity == 0u) {
        new_capacity = t->n_dense_rows < BEAM_SPARSE_INITIAL_CAPACITY
            ? t->n_dense_rows : BEAM_SPARSE_INITIAL_CAPACITY;
    }
    while (new_capacity < needed) {
        if (new_capacity > UINT32_MAX / 2u) {
            new_capacity = needed;
            break;
        }
        new_capacity *= 2u;
        if (new_capacity > t->n_dense_rows)
            new_capacity = t->n_dense_rows;
    }
    if (new_capacity < needed) return false;

    size_t value_bytes;
    if (!sparse_value_bytes(new_capacity, t->stride, &value_bytes)) return false;

    double *new_values = (double *)malloc(value_bytes);
    if (!new_values) return false;

    /* Mutable tables only -- `t->row_ids` is NULL on a finalized one, and
     * that pointer is the discriminator every site here reads, rather than
     * `mutable_rows`. See beam_sparse_init for why a finalized table has no
     * map to grow. */
    uint32_t *new_row_ids = NULL;
    if (t->row_ids) {
        size_t row_id_bytes;
        if (!sparse_array_bytes((size_t)new_capacity,
                                sizeof(*new_row_ids), &row_id_bytes)) {
            free(new_values);
            return false;
        }
        new_row_ids = (uint32_t *)malloc(row_id_bytes);
        if (!new_row_ids) {
            free(new_values);
            return false;
        }
    }

    if (occupied != 0u) {
        size_t occupied_bytes;
        if (!sparse_value_bytes(occupied, t->stride, &occupied_bytes)) {
            free(new_row_ids);
            free(new_values);
            return false;
        }
        memcpy(new_values, t->values[part], occupied_bytes);
        if (t->row_ids && t->row_ids[part])
            memcpy(new_row_ids, t->row_ids[part],
                   (size_t)occupied * sizeof(*new_row_ids));
    }

    free(t->values[part]);
    t->values[part] = new_values;
    if (!t->mutable_rows && part == 0u) t->values0 = new_values;
    if (t->row_ids) {
        free(t->row_ids[part]);
        t->row_ids[part] = new_row_ids;
    }
    t->capacity[part] = new_capacity;
    return true;
}

BeamError beam_sparse_init(BeamSparseRows *t, uint32_t n_dense_rows,
                              uint32_t n_parts, uint32_t stride,
                              bool mutable_rows) {
    if (!t) return BEAM_ENULL("sparse table");

    memset(t, 0, sizeof(*t));
    if (n_parts == 0u) {
        return BEAM_ERANGE("%s: n_parts must be at least one", __func__);
    }
    /* Without row_ids there is no way to recover a moved row's partition.
     * Finalized tables deliberately have one payload, so reject an ambiguous
     * layout instead of allowing a wrong-part lookup to alias another row. */
    if (!mutable_rows && n_parts != 1u) {
        return BEAM_ERANGE(
            "%s: immutable sparse tables must have exactly one partition",
            __func__);
    }

    if (stride == 0u) {
        return BEAM_ERANGE("%s: stride must be at least one", __func__);
    }

    t->n_dense_rows = n_dense_rows;
    t->n_parts = n_parts;
    t->stride = stride;
    t->mutable_rows = mutable_rows;

    if (n_dense_rows != 0u) {
        size_t index_bytes;
        if (!sparse_array_bytes((size_t)n_dense_rows, sizeof(*t->index),
                                &index_bytes))
            goto alloc_failed;
        t->index = (uint32_t *)malloc(index_bytes);
        if (!t->index) goto alloc_failed;
        for (uint32_t row = 0; row < n_dense_rows; ++row)
            t->index[row] = BEAM_ROW_MISSING;
    }

    size_t ignored;
    if (!sparse_array_bytes((size_t)n_parts, sizeof(*t->values), &ignored) ||
        !sparse_array_bytes((size_t)n_parts, sizeof(*t->size), &ignored) ||
        !sparse_array_bytes((size_t)n_parts, sizeof(*t->capacity), &ignored) ||
        !sparse_array_bytes((size_t)n_parts, sizeof(*t->row_ids), &ignored))
        goto alloc_failed;

    t->values = (double **)calloc((size_t)n_parts, sizeof(*t->values));
    t->size = (uint32_t *)calloc((size_t)n_parts, sizeof(*t->size));
    t->capacity = (uint32_t *)calloc((size_t)n_parts,
                                     sizeof(*t->capacity));
    /*
     * Mutable tables only.
     *
     * A finalized table used to carry one so a caller could list its occupied
     * rows without scanning all n_dense_rows. Exactly one caller ever did --
     * `beam_layer_max` -- and it now walks the directory slice holding the
     * depth bucket it wants, which reaches the same rows in the same order
     * while reading 194,481 entries where the map made it read 1,846,623.
     *
     * So on a finalized table the map is four bytes per occupied row that
     * nothing loads: 16.32 MiB across the foundation's six of them. A mutable
     * table still needs its own, to recover a moved row's identity after an
     * erase and to make a wrong-partition lookup a miss rather than an alias.
     */
    if (mutable_rows) {
        t->row_ids = (uint32_t **)calloc((size_t)n_parts,
                                             sizeof(*t->row_ids));
        if (!t->row_ids) goto alloc_failed;
    }

    if (!t->values || !t->size || !t->capacity)
        goto alloc_failed;
    return BEAM_OK;

alloc_failed:
    beam_sparse_destroy(t);
    return BEAM_EALLOC("a sparse BEAM row table");
}

void beam_sparse_destroy(BeamSparseRows *t) {
    if (!t) return;
    if (t->values) {
        for (uint32_t part = 0; part < t->n_parts; ++part)
            free(t->values[part]);
    }
    if (t->row_ids) {
        for (uint32_t part = 0; part < t->n_parts; ++part)
            free(t->row_ids[part]);
    }
    free(t->index);
    free(t->rank);
    free(t->values);
    free(t->row_ids);
    free(t->size);
    free(t->capacity);
    memset(t, 0, sizeof(*t));
}

/**
 * The slot a sealed directory gives `row`, or BEAM_ROW_MISSING.
 *
 * The same arithmetic as the inline finalized lookup in the header; kept here
 * for the validating path, which is not hot and does not justify a second
 * copy in every translation unit.
 */
static uint32_t sparse_rank_slot(const BeamSparseRows *t, uint32_t row) {
    const BeamRankBlock *b = &t->rank[row >> 8];
    const uint32_t ofs  = row & 255u;
    const uint32_t w    = ofs >> 6;
    const uint64_t word = b->bits[w];
    const uint64_t bit  = 1ull << (ofs & 63u);
    if (!(word & bit)) return BEAM_ROW_MISSING;
    const uint32_t sub = w ? ((b->sub >> (9u * (w - 1u))) & 511u) : 0u;
    return b->rank + sub + (uint32_t)__builtin_popcountll(word & (bit - 1ull));
}

const double *beam_sparse_row(const BeamSparseRows *t, uint32_t row,
                                  uint32_t part) {
    if (!t || row >= t->n_dense_rows || part >= t->n_parts ||
        (!t->index && !t->rank) || !t->values || !t->size || !t->capacity)
        return NULL;

    const uint32_t slot = t->rank ? sparse_rank_slot(t, row) : t->index[row];
    if (slot == BEAM_ROW_MISSING || slot >= t->size[part] ||
        slot >= t->capacity[part] || !t->values[part])
        return NULL;

    /* Mutable tables may have several payloads.  The reverse map makes a
     * caller supplying the wrong partition a clean miss rather than an alias
     * to whichever row happens to occupy the same slot there. */
    if (t->mutable_rows) {
        if (!t->row_ids || !t->row_ids[part] ||
            t->row_ids[part][slot] != row)
            return NULL;
    } else if (part != 0u) {
        return NULL;
    }

    return t->values[part] + (size_t)slot * t->stride;
}

double *beam_sparse_row_mut(BeamSparseRows *t, uint32_t row,
                                uint32_t part, bool create) {
    if (!t || row >= t->n_dense_rows || part >= t->n_parts || !t->index ||
        !t->values || !t->size || !t->capacity ||
        (t->mutable_rows && !t->row_ids) ||
        (!t->mutable_rows && part != 0u))
        return NULL;

    const uint32_t old_slot = t->index[row];
    if (old_slot != BEAM_ROW_MISSING) {
        if (old_slot >= t->size[part] || old_slot >= t->capacity[part] ||
            !t->values[part])
            return NULL;
        if (t->mutable_rows &&
            (!t->row_ids[part] || t->row_ids[part][old_slot] != row))
            return NULL;
        return t->values[part] + (size_t)old_slot * t->stride;
    }
    if (!create) return NULL;

    const uint32_t slot = t->size[part];
    if (slot == UINT32_MAX || slot >= t->n_dense_rows) return NULL;
    if (!sparse_grow_partition(t, part, slot + 1u)) {
        beam_set_error(BEAM_ERR_ALLOC,
                      "%s: out of memory growing sparse BEAM row payload",
                      __func__);
        return NULL;
    }

    double *result = t->values[part] + (size_t)slot * t->stride;
    memset(result, 0, (size_t)t->stride * sizeof(*result));
    if (t->row_ids && t->row_ids[part]) t->row_ids[part][slot] = row;
    t->size[part] = slot + 1u;
    t->index[row] = slot;
    return result;
}

bool beam_sparse_erase(BeamSparseRows *t, uint32_t row, uint32_t part) {
    if (!t || !t->mutable_rows || row >= t->n_dense_rows ||
        part >= t->n_parts || !t->index || !t->values || !t->row_ids ||
        !t->size || !t->capacity || !t->values[part] || !t->row_ids[part])
        return false;

    const uint32_t slot = t->index[row];
    const uint32_t occupied = t->size[part];
    if (slot == BEAM_ROW_MISSING || occupied == 0u ||
        occupied > t->capacity[part] || slot >= occupied ||
        t->row_ids[part][slot] != row)
        return false;

    const uint32_t last = occupied - 1u;
    if (slot != last) {
        const uint32_t moved_row = t->row_ids[part][last];
        if (moved_row >= t->n_dense_rows || moved_row == row ||
            t->index[moved_row] != last)
            return false;
        memcpy(t->values[part] + (size_t)slot * t->stride,
               t->values[part] + (size_t)last * t->stride,
               (size_t)t->stride * sizeof(double));
        t->row_ids[part][slot] = moved_row;
        t->index[moved_row] = slot;
    }

    memset(t->values[part] + (size_t)last * t->stride, 0,
           (size_t)t->stride * sizeof(double));
    t->row_ids[part][last] = BEAM_ROW_MISSING;
    t->index[row] = BEAM_ROW_MISSING;
    t->size[part] = last;
    return true;
}

void beam_sparse_shrink(BeamSparseRows *t) {
    if (!t || !t->values || !t->size || !t->capacity) return;

    for (uint32_t part = 0; part < t->n_parts; ++part) {
        const uint32_t occupied = t->size[part];
        if (occupied > t->capacity[part]) continue;
        if (occupied == 0u) {
            free(t->values[part]);
            t->values[part] = NULL;
            if (part == 0u) t->values0 = NULL;
            if (t->row_ids) {
                free(t->row_ids[part]);
                t->row_ids[part] = NULL;
            }
            t->capacity[part] = 0u;
            continue;
        }
        if (occupied == t->capacity[part] || !t->values[part]) continue;
        if (t->mutable_rows && (!t->row_ids || !t->row_ids[part])) continue;

        size_t value_bytes;
        if (!sparse_value_bytes(occupied, t->stride, &value_bytes)) continue;
        double *new_values = (double *)malloc(value_bytes);
        if (!new_values) continue;

        /* Keyed on whether the partition HAS a row map, never on the kind of
         * table -- the two have disagreed twice now, in both directions.
         * Allocating only for mutable while copying whenever a map existed
         * memcpy'd into NULL and segfaulted, back when finalized tables
         * carried a map; they no longer do, which would have made the mirror
         * of that assumption wrong instead. The pointer is the only fact that
         * has stayed true. */
        const bool has_map = (t->row_ids && t->row_ids[part]);
        uint32_t *new_row_ids = NULL;
        if (has_map) {
            new_row_ids = (uint32_t *)malloc((size_t)occupied
                                             * sizeof(*new_row_ids));
            if (!new_row_ids) {
                free(new_values);
                continue;
            }
        }

        memcpy(new_values, t->values[part], value_bytes);
        if (has_map)
            memcpy(new_row_ids, t->row_ids[part],
                   (size_t)occupied * sizeof(*new_row_ids));

        free(t->values[part]);
        t->values[part] = new_values;
        if (!t->mutable_rows && part == 0u) t->values0 = new_values;
        if (has_map) {
            free(t->row_ids[part]);
            t->row_ids[part] = new_row_ids;
        }
        t->capacity[part] = occupied;
    }
}

BeamError beam_sparse_seal_rank(BeamSparseRows *t) {
    if (!t) return BEAM_ENULL("sparse table");
    /* Sealing is an optimisation. A table that cannot take it is not an
     * error; it simply keeps the directory it has. */
    if (t->mutable_rows || t->rank || !t->index || t->n_dense_rows == 0u)
        return BEAM_OK;

    const uint32_t occupied = t->size ? t->size[0] : 0u;
    const uint32_t n_blocks = beam_rank_blocks(t->n_dense_rows);
    BeamRankBlock *blocks = (BeamRankBlock *)calloc((size_t)n_blocks,
                                                    sizeof(*blocks));
    if (!blocks) return BEAM_EALLOC("a rank directory");

    /*
     * Build the bitmap and the prefix counts in one ascending pass, and
     * CHECK the invariant the whole form rests on rather than assuming it:
     * a finalized table hands out slots in ascending row order, so a row's
     * slot must equal the number of occupied rows before it. If that is ever
     * untrue the rank directory would silently return another row's values,
     * which is the worst failure this file could have, so it is worth one
     * comparison per occupied row to refuse instead.
     */
    uint32_t seen = 0;
    for (uint32_t blk = 0; blk < n_blocks; ++blk) {
        blocks[blk].rank = seen;
        uint32_t within = 0;
        for (uint32_t w = 0; w < 4u; ++w) {
            if (w) {
                /* At most 192 before word 3, so nine bits are exact. */
                blocks[blk].sub |= (within & 511u) << (9u * (w - 1u));
            }
            const uint32_t base = blk * BEAM_RANK_BLOCK_ROWS + w * 64u;
            uint64_t bits = 0;
            for (uint32_t k = 0; k < 64u; ++k) {
                const uint32_t row = base + k;
                if (row >= t->n_dense_rows) break;
                const uint32_t slot = t->index[row];
                if (slot == BEAM_ROW_MISSING) continue;
                if (slot != seen + within) {   /* not ascending: refuse */
                    free(blocks);
                    return BEAM_ERANGE(
                        "%s: finalized slot %u for row %u is not the %u rows "
                        "before it; the table was not filled in ascending "
                        "order and cannot be sealed",
                        __func__, slot, row, seen + within);
                }
                bits |= 1ull << k;
                within++;
            }
            blocks[blk].bits[w] = bits;
        }
        seen += within;
    }
    if (seen != occupied) {
        free(blocks);
        return BEAM_ERANGE("%s: directory holds %u rows but the payload has %u",
                           __func__, seen, occupied);
    }

    t->rank = blocks;
    free(t->index);
    t->index = NULL;
    return BEAM_OK;
}

static uint64_t sparse_add_saturating(uint64_t a, uint64_t b) {
    return UINT64_MAX - a < b ? UINT64_MAX : a + b;
}

static uint64_t sparse_mul_saturating(uint64_t a, uint64_t b) {
    return a != 0u && b > UINT64_MAX / a ? UINT64_MAX : a * b;
}

uint64_t beam_sparse_memory_bytes(const BeamSparseRows *t) {
    if (!t) return 0u;

    uint64_t bytes = 0u;
    if (t->index)
        bytes = sparse_add_saturating(
            bytes, sparse_mul_saturating(t->n_dense_rows,
                                         sizeof(*t->index)));
    if (t->rank)
        bytes = sparse_add_saturating(
            bytes, sparse_mul_saturating(beam_rank_blocks(t->n_dense_rows),
                                         sizeof(*t->rank)));
    if (t->values)
        bytes = sparse_add_saturating(
            bytes, sparse_mul_saturating(t->n_parts, sizeof(*t->values)));
    if (t->row_ids)
        bytes = sparse_add_saturating(
            bytes, sparse_mul_saturating(t->n_parts, sizeof(*t->row_ids)));
    if (t->size)
        bytes = sparse_add_saturating(
            bytes, sparse_mul_saturating(t->n_parts, sizeof(*t->size)));
    if (t->capacity)
        bytes = sparse_add_saturating(
            bytes, sparse_mul_saturating(t->n_parts, sizeof(*t->capacity)));

    if (t->capacity) {
        for (uint32_t part = 0; part < t->n_parts; ++part) {
            const uint64_t capacity = t->capacity[part];
            if (t->values && t->values[part]) {
                bytes = sparse_add_saturating(
                    bytes, sparse_mul_saturating(
                               capacity,
                               (uint64_t)t->stride * sizeof(double)));
            }
            if (t->row_ids && t->row_ids[part]) {
                bytes = sparse_add_saturating(
                    bytes, sparse_mul_saturating(capacity,
                                                 sizeof(uint32_t)));
            }
        }
    }
    return bytes;
}

uint64_t beam_sparse_n_rows(const BeamSparseRows *t) {
    if (!t || !t->size) return 0u;
    uint64_t rows = 0u;
    for (uint32_t part = 0; part < t->n_parts; ++part)
        rows += t->size[part];
    return rows;
}
