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

static bool sparse_value_bytes(uint32_t rows, size_t *out) {
    const size_t row_bytes = (size_t)BEAM_RESIDUES * sizeof(double);
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
    if (!sparse_value_bytes(new_capacity, &value_bytes)) return false;

    double *new_values = (double *)malloc(value_bytes);
    if (!new_values) return false;

    /* Recorded for IMMUTABLE tables as well as mutable ones. A finalized
     * table's rows are added in ascending dense-row order, so the map comes
     * out sorted for free -- and without it the only way to learn which rows
     * a sparse table holds is to scan all n_dense_rows, which is the O(S)
     * cost the sparse layout exists to avoid. Four bytes per occupied row. */
    uint32_t *new_row_ids = NULL;
    {
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
        if (!sparse_value_bytes(occupied, &occupied_bytes)) {
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
    {
        free(t->row_ids[part]);
        t->row_ids[part] = new_row_ids;
    }
    t->capacity[part] = new_capacity;
    return true;
}

BeamError beam_sparse_init(BeamSparseRows *t, uint32_t n_dense_rows,
                              uint32_t n_parts, bool mutable_rows) {
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

    t->n_dense_rows = n_dense_rows;
    t->n_parts = n_parts;
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
    /* Allocated for immutable tables too: their slot -> row map is what lets
     * a caller list the rows a finalized table holds without scanning all
     * n_dense_rows. */
    t->row_ids = (uint32_t **)calloc((size_t)n_parts,
                                         sizeof(*t->row_ids));

    if (!t->values || !t->size || !t->capacity ||
        (mutable_rows && !t->row_ids))
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
    free(t->values);
    free(t->row_ids);
    free(t->size);
    free(t->capacity);
    memset(t, 0, sizeof(*t));
}

const double *beam_sparse_row(const BeamSparseRows *t, uint32_t row,
                                  uint32_t part) {
    if (!t || row >= t->n_dense_rows || part >= t->n_parts || !t->index ||
        !t->values || !t->size || !t->capacity)
        return NULL;

    const uint32_t slot = t->index[row];
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

    return t->values[part] + (size_t)slot * BEAM_RESIDUES;
}

double *beam_sparse_row_mut(BeamSparseRows *t, uint32_t row,
                                uint32_t part, bool create) {
    if (!t || row >= t->n_dense_rows || part >= t->n_parts || !t->index ||
        !t->values || !t->size || !t->capacity || !t->row_ids ||
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
        return t->values[part] + (size_t)old_slot * BEAM_RESIDUES;
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

    double *result = t->values[part] + (size_t)slot * BEAM_RESIDUES;
    memset(result, 0, (size_t)BEAM_RESIDUES * sizeof(*result));
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
        memcpy(t->values[part] + (size_t)slot * BEAM_RESIDUES,
               t->values[part] + (size_t)last * BEAM_RESIDUES,
               (size_t)BEAM_RESIDUES * sizeof(double));
        t->row_ids[part][slot] = moved_row;
        t->index[moved_row] = slot;
    }

    memset(t->values[part] + (size_t)last * BEAM_RESIDUES, 0,
           (size_t)BEAM_RESIDUES * sizeof(double));
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
        if (!sparse_value_bytes(occupied, &value_bytes)) continue;
        double *new_values = (double *)malloc(value_bytes);
        if (!new_values) continue;

        /* Keyed on whether the partition HAS a row map, not on whether the
         * table is mutable: immutable tables carry one now too. Allocating
         * only for mutable while copying whenever the map exists memcpy'd
         * into NULL and segfaulted. */
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
        if (has_map) {
            free(t->row_ids[part]);
            t->row_ids[part] = new_row_ids;
        }
        t->capacity[part] = occupied;
    }
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
                               (uint64_t)BEAM_RESIDUES * sizeof(double)));
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


/**
 * The dense row ids this table holds, in ASCENDING order, for one partition.
 *
 * Finalized tables are filled in ascending dense-row order, so the map is
 * already sorted and callers can walk it directly. Mutable count tables are
 * filled in observation order and are NOT sorted; a caller needing order must
 * sort a copy.
 *
 * Returns NULL and sets `*n` to zero when the partition holds nothing.
 */
const uint32_t *beam_sparse_rows(const BeamSparseRows *t, uint32_t part,
                                     uint32_t *n) {
    if (n) *n = 0u;
    if (!t || part >= t->n_parts || !t->row_ids || !t->row_ids[part] || !t->size)
        return NULL;
    if (n) *n = t->size[part];
    return t->row_ids[part];
}
