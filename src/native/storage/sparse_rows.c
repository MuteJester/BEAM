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
 * Install buffers of exactly `new_capacity` rows on one partition, keeping
 * the occupied prefix.
 *
 * realloc cannot be used here: the values may move successfully and the
 * matching row-id allocation then fail, leaving the table describing an array
 * it no longer owns. Both replacements are allocated and filled first, and
 * only then committed, so a failure anywhere leaves the partition exactly as
 * it was.
 *
 * Separated from the decision of WHAT capacity to install, because there are
 * two callers with opposite information: growth is discovered a row at a time
 * and doubles, while `beam_sparse_reserve` is told the answer up front.
 */
static bool sparse_set_capacity(BeamSparseRows *t, uint32_t part,
                                uint32_t new_capacity) {
    const uint32_t occupied = t->size[part];
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

/** Shared entry checks for both capacity paths. */
static bool sparse_capacity_precheck(const BeamSparseRows *t, uint32_t part,
                                     uint32_t needed) {
    if (!t || part >= t->n_parts || !t->values || !t->size ||
        !t->capacity || needed == 0u || needed > t->n_dense_rows)
        return false;
    const uint32_t occupied = t->size[part];
    if (occupied > t->capacity[part]) return false;
    if (occupied != 0u && !t->values[part]) return false;
    if (t->mutable_rows &&
        (!t->row_ids || (occupied != 0u && !t->row_ids[part])))
        return false;
    return true;
}

/**
 * Grow a partition to hold at least `needed` rows, doubling.
 *
 * Doubling is what an incremental filler needs: it discovers rows one at a
 * time and cannot be told the total. A caller that DOES know the total should
 * say so with `beam_sparse_reserve` and avoid both the slack and the copies.
 */
static bool sparse_grow_partition(BeamSparseRows *t, uint32_t part,
                                  uint32_t needed) {
    if (!sparse_capacity_precheck(t, part, needed)) return false;
    const uint32_t old_capacity = t->capacity[part];
    if (old_capacity >= needed) return true;

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
    return sparse_set_capacity(t, part, new_capacity);
}

BeamError beam_sparse_reserve(BeamSparseRows *t, uint32_t part,
                              uint32_t rows) {
    if (rows == 0u) return BEAM_OK;
    if (!sparse_capacity_precheck(t, part, rows))
        return BEAM_ERANGE("%s: cannot reserve %u rows of %u", __func__,
                           rows, t ? t->n_dense_rows : 0u);
    if (t->capacity[part] >= rows) return BEAM_OK;   /* already big enough */
    if (!sparse_set_capacity(t, part, rows))
        return BEAM_EALLOC("an exact-capacity sparse partition");
    return BEAM_OK;
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
    free(t->pool_id);
    free(t->cell_mask); free(t->cell_start); free(t->cell_index); free(t->dict);
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

    uint32_t slot = t->rank ? sparse_rank_slot(t, row) : t->index[row];
    if (slot == BEAM_ROW_MISSING || slot >= t->size[part] || !t->values[part])
        return NULL;
    /* The directory names a dense position; a pooled table then maps that to
     * the row several positions share. Both are range-checked, because a
     * corrupt pool entry would otherwise read another row's values. */
    if (t->pool_id) {
        slot = t->pool_id[slot];
        if (slot >= t->n_pool) return NULL;
    }
    if (slot >= t->capacity[part]) return NULL;

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

/** 64-bit mix, for hashing a row's bytes. Bit patterns only: two rows are
 *  interchangeable exactly when all their bytes match. */
static inline uint64_t sparse_mix(uint64_t x) {
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL;
    return x ^ (x >> 33);
}

BeamError beam_sparse_intern_rows(BeamSparseRows *t) {
    if (!t) return BEAM_ENULL("sparse table");
    /* Like sealing, this is an optimisation: a table that cannot take it
     * keeps what it has and reports OK. */
    if (t->mutable_rows || t->pool_id || !t->values || !t->values[0]
        || !t->size || !t->capacity)
        return BEAM_OK;
    const uint32_t n = t->size[0];
    const uint32_t R = t->stride;
    if (n < 2u || R == 0u) return BEAM_OK;
    const size_t RB = (size_t)R * sizeof(double);

    size_t slots = 1u;
    while (slots < (size_t)n * 2u) {
        if (slots > SIZE_MAX / 2u) return BEAM_OK;      /* absurdly large */
        slots <<= 1;
    }
    uint32_t *ht = (uint32_t *)malloc(slots * sizeof(*ht));
    uint32_t *pool_id = (uint32_t *)malloc((size_t)n * sizeof(*pool_id));
    if (!ht || !pool_id) {
        free(ht); free(pool_id);
        return BEAM_EALLOC("the row-interning scratch");
    }
    memset(ht, 0xFF, slots * sizeof(*ht));              /* BEAM_ROW_MISSING */

    /*
     * One ascending pass. A slot whose bytes have been seen before takes the
     * earlier row's pool index; otherwise it opens a new one. Pool indices
     * are therefore handed out in order of FIRST appearance, which is what
     * lets the payload below be rebuilt in one more ascending pass with no
     * second bookkeeping array.
     *
     * The hash only chooses where to look. Membership is decided by memcmp
     * over every byte, so a collision costs a comparison and can never pool
     * two rows that differ.
     */
    const double *src = t->values[0];
    uint32_t n_pool = 0;
    for (uint32_t k = 0; k < n; ++k) {
        const double *row = src + (size_t)k * R;
        uint64_t h = 1469598103934665603ULL;
        for (size_t b = 0; b < RB; b += sizeof(uint64_t)) {
            uint64_t word;
            memcpy(&word, (const char *)row + b, sizeof word);
            h = sparse_mix(h ^ word);
        }
        size_t j = (size_t)h & (slots - 1u);
        for (;;) {
            const uint32_t cand = ht[j];
            if (cand == BEAM_ROW_MISSING) {
                ht[j] = k;
                pool_id[k] = n_pool++;
                break;
            }
            if (memcmp(src + (size_t)cand * R, row, RB) == 0) {
                pool_id[k] = pool_id[cand];
                break;
            }
            j = (j + 1u) & (slots - 1u);
        }
    }
    free(ht);

    /* Not enough repetition to pay for the extra load. Leave it alone. */
    if ((uint64_t)(n - n_pool) * BEAM_INTERN_MIN_DUPLICATE_DENOM
        < (uint64_t)n * BEAM_INTERN_MIN_DUPLICATE_NUM) {
        free(pool_id);
        return BEAM_OK;
    }

    double *pool = (double *)malloc((size_t)n_pool * RB);
    if (!pool) { free(pool_id); return BEAM_EALLOC("the interned row pool"); }
    uint32_t filled = 0;
    for (uint32_t k = 0; k < n; ++k)
        if (pool_id[k] == filled) {          /* the first time this row appears */
            memcpy(pool + (size_t)filled * R, src + (size_t)k * R, RB);
            filled++;
        }
    if (filled != n_pool) {                  /* cannot happen; refuse if it does */
        free(pool); free(pool_id);
        return BEAM_ERANGE("%s: filled %u pool rows for %u distinct",
                           __func__, filled, n_pool);
    }

    free(t->values[0]);
    t->values[0] = pool;
    t->values0 = pool;
    t->capacity[0] = n_pool;
    t->n_pool = n_pool;

    /*
     * A table that still has a direct directory needs no second array and no
     * second load: its entry per dense row can name the pooled row outright.
     * Only a sealed table needs the map, because rank(row) is inherently the
     * count of rows before this one and cannot be renumbered.
     */
    if (t->index) {
        for (uint32_t row = 0; row < t->n_dense_rows; ++row) {
            const uint32_t slot = t->index[row];
            if (slot != BEAM_ROW_MISSING) t->index[row] = pool_id[slot];
        }
        free(pool_id);
    } else {
        t->pool_id = pool_id;
    }
    return BEAM_OK;
}

BeamError beam_sparse_seal_rank(BeamSparseRows *t) {
    if (!t) return BEAM_ENULL("sparse table");
    /*
     * Sealing is an optimisation. A table that cannot take it is not an
     * error; it simply keeps the directory it has.
     *
     * `n_pool` covers an already-interned table, and covers it whichever way
     * the pool was attached -- a direct table folds the pool into `index`,
     * which is exactly the ascending-slot property this form needs, so the
     * check below would otherwise refuse with a confusing message rather than
     * decline. Interning after sealing is the supported order; see
     * finalize_sparse_table.
     */
    if (t->mutable_rows || t->rank || t->n_pool || !t->index
        || t->n_dense_rows == 0u)
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
    if (t->cell_mask) {
        bytes = sparse_add_saturating(
            bytes, sparse_mul_saturating(t->n_cell_rows, sizeof(*t->cell_mask)));
        bytes = sparse_add_saturating(
            bytes, sparse_mul_saturating((uint64_t)t->n_cell_rows + 1u,
                                         sizeof(*t->cell_start)));
        bytes = sparse_add_saturating(
            bytes, sparse_mul_saturating(t->cell_start[t->n_cell_rows],
                                         sizeof(*t->cell_index)));
        bytes = sparse_add_saturating(
            bytes, sparse_mul_saturating(t->n_dict, sizeof(*t->dict)));
    }
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
            if (part == 0u && t->pool_id)
                bytes = sparse_add_saturating(
                    bytes, sparse_mul_saturating(t->size[0], sizeof(uint32_t)));
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
