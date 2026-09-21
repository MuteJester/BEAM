/**
 * @file compaction.c
 * @brief The four transforms a finalized sparse table gets, in the one order.
 *
 * Sealing the directory to rank form, interning byte-identical rows, packing
 * cells into a value dictionary, and transposing a backward table plane-major.
 * Each is implemented next to the representation it produces --
 * `storage/sparse_rows.c` and `storage/packed_rows.c` -- and each was CALLED
 * from `model/finalize.c`, forty lines apart, with the order that makes them
 * work stated in prose between them.
 *
 * ## Why the order is load-bearing
 *
 * Seal before intern: sealing requires each row's slot to be the count of rows
 * before it, and interning is precisely what breaks that -- it points several
 * slots at one row. The other way round the seal refuses, correctly, and the
 * directory stays 40.06 MiB per generation at order 4.
 *
 * Pack after both: it is the step that gives up the contiguous twenty-double
 * row. Sealing and interning still address rows; after packing a row is a mask
 * and a run of indices.
 *
 * ## Pack and transpose are ALTERNATIVES, and pack wins
 *
 * This is the fact the old arrangement did not state anywhere. The transpose
 * declines for a packed table -- there is no array of rows left to transpose --
 * so running pack first is a decision that pack takes priority, not an
 * ordering detail. Measured across the test fixtures: every backward table of
 * every model the suite builds comes out PACKED and never transposed.
 *
 * The transpose is not dead. It applies exactly where packing declines, which
 * is when the value dictionary would exceed `BEAM_PACK_MAX_DICT` -- the
 * foundation model, whose cells take more distinct values than a two-byte
 * index can name. So the two split the world at corpus scale: repertoire-sized
 * models pack, foundation-sized ones transpose. Nothing said so before, and a
 * test that asserted "finalize never transposes a forward table" was passing
 * on a fixture where nothing was transposed at all.
 *
 * ## Why every step still returns BEAM_OK when it declines
 *
 * A table that cannot take a transform keeps what it has, and for a single
 * call that is right: none of these is required for correctness, only for
 * size. What was missing is a way to find out. `BeamCompactionApplied` is
 * that: the caller learns which four happened, and a test can assert the
 * answer for a fixture instead of assuming it.
 */
#include "beam/internal.h"

BeamError beam_sparse_compact_finalized(BeamSparseRows *t, uint32_t n_dense,
                                        bool backward,
                                        BeamCompactionApplied *applied) {
    if (!t) return BEAM_ENULL("sparse table");
    BeamCompactionApplied got;
    memset(&got, 0, sizeof(got));

    /*
     * A table almost none of whose rows exist spends nearly all of itself on
     * a directory of absences: 40.06 MiB per generation at order 4 whether the
     * corpus was nine hundred million sequences or nine hundred. Below the
     * measured threshold, swap that directory for the rank form -- see
     * BEAM_RANK_SEAL_MAX_OCCUPANCY_DENOM for what was measured and why the
     * line is where it is.
     *
     * Sealing is exact: the same rows, the same slots, the same doubles. The
     * caller runs this before `beam_compute_log_z` reads these tables, so the
     * partition functions are computed against the representation the model
     * will actually keep rather than against one it is about to discard.
     *
     * The occupancy test lives here rather than at the call site because the
     * threshold it compares against is defined in `sparse_rows.h`: a constant
     * and its only use belong in one subsystem.
     */
    const uint32_t occupied = t->size ? t->size[0] : 0u;
    if ((uint64_t)occupied * BEAM_RANK_SEAL_MAX_OCCUPANCY_DENOM
            <= (uint64_t)n_dense) {
        const BeamError e = beam_sparse_seal_rank(t);
        if (e != BEAM_OK) return e;
        got.sealed = t->rank != NULL;
    }

    /*
     * Then share storage between rows that are already byte-identical. A
     * context seen once normalises to one residue at 0.0 and nineteen at
     * -inf, and only twenty such rows can exist, so a repertoire model is
     * about half copies of twenty rows: 73-79% of a donor's rows repeat.
     */
    const BeamError ie = beam_sparse_intern_rows(t);
    if (ie != BEAM_OK) return ie;
    got.interned = t->n_pool != 0u;

    /*
     * Then replace the values themselves with indices into the distinct ones.
     * Every cell here is `log(c / t)` for integers c <= t, and a repertoire's
     * counts are small, so after interning has removed the repeated ROWS, 95%
     * of the remaining cells are still a value stored somewhere else:
     * 1,175,396 cells take 59,008 distinct values on a 197,139-sequence donor.
     */
    const BeamError pe = beam_sparse_pack_cells(t);
    if (pe != BEAM_OK) return pe;
    got.packed = t->cell_mask != NULL;

    /*
     * A backward table that still holds plain rows is laid out by residue.
     *
     * Nothing reads a backward row whole -- `beam_log_row_into` is called only
     * on the three forward rows -- and the backward half of the edge kernel
     * reads ONE residue from each of twenty rows whose dense ids are
     * consecutive. Measured on the foundation: 78,538 of 148,176 such tiles
     * have all twenty rows present, they hold 85.2% of the occupied rows, and
     * in every one of them the slots are consecutive too. By row that is fifty
     * cache lines to use a hundred and sixty bytes; by residue those bytes are
     * contiguous.
     *
     * Declined for a packed table, which is why `got.packed` and
     * `got.transposed` are never both true, and for the forward tables, which
     * are read whole and want the rows exactly as they are.
     */
    if (backward) {
        const BeamError te = beam_sparse_plane_transpose(t);
        if (te != BEAM_OK) return te;
        got.transposed = t->plane_step > 1u;
    }

    if (applied) *applied = got;
    return BEAM_OK;
}
