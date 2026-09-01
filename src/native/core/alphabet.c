/**
 * @file alphabet.c
 * @brief The alphabet registry.
 *
 * Data lives in alphabet_aa.c and alphabet_nt.c, one file per alphabet, so
 * that adding an alphabet is adding a file rather than editing a switch. This
 * file only maps an identifier or a name onto one of them.
 */
#include <string.h>

#include "beam/common.h"
#include "beam/model.h"
#include "beam/alphabet.h"

/* ── Registry ────────────────────────────────────────────────── */

/* Indexed by BeamAlphabet, so a new alphabet is a new entry here and a new
 * enumerator. The assertion below is what keeps those two in step. */
static const BeamAlphabetInfo *const registry[BEAM_ALPHABET_COUNT] = {
    [BEAM_ALPHABET_AA] = &beam_alphabet_aa,
    [BEAM_ALPHABET_NT] = &beam_alphabet_nt,
};

const BeamAlphabetInfo *beam_alphabet_info(BeamAlphabet id) {
    if ((unsigned)id >= BEAM_ALPHABET_COUNT) return NULL;
    return registry[id];
}

const BeamAlphabetInfo *beam_alphabet_by_name(const char *name) {
    if (!name) return NULL;
    for (unsigned i = 0; i < BEAM_ALPHABET_COUNT; ++i)
        if (strcmp(name, registry[i]->name) == 0) return registry[i];
    return NULL;
}
