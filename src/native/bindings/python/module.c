/**
 * @file module.c
 * @brief The method table and the single module initialiser.
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <string.h>
#include <math.h>
#include <inttypes.h>
#include <limits.h>
#include <stdbool.h>

#include "beam/common.h"
#include "beam/model.h"
#include "beam/publicness.h"

/* Capsule tag. Carried over verbatim so a model pickled or passed between the
 * two packages during the transition is still recognised; it is renamed with
 * everything else in the rename step. */
#include "beam/python_bindings.h"

const char *BEAM_CAPSULE_NAME = "BeamModel";


static PyMethodDef module_methods[] = {
    {"publicness_moments",      py_publicness_moments,                 METH_VARARGS, NULL},
    {"publicness_pgf",          py_publicness_pgf,                     METH_VARARGS, NULL},
    {"publicness_accumulate",   py_publicness_accumulate,              METH_VARARGS, NULL},
    {"publicness_accumulate_grouped", py_publicness_accumulate_grouped, METH_VARARGS, NULL},
    {"build",              py_build,                         METH_VARARGS, NULL},
    {"new",                py_new,                           METH_VARARGS, NULL},
    {"observe",            py_observe,                       METH_VARARGS, NULL},
    {"finalize",           py_finalize,                      METH_O, NULL},
    {"is_finalized",       py_is_finalized,                  METH_O, NULL},
    {"set_threads",        py_set_threads,                   METH_O, NULL},
    {"threads",            py_threads,                       METH_NOARGS, NULL},
    {"effective_threads",  py_effective_threads,             METH_NOARGS, NULL},
    {"set_memory_budget",  py_set_memory_budget,             METH_O, NULL},
    {"memory_budget",      py_memory_budget,                 METH_NOARGS, NULL},
    {"spectrum_pair",      py_spectrum_pair,                 METH_VARARGS, NULL},
    {"spectra_by_length",  py_spectra_by_length,             METH_VARARGS, NULL},
    {"length_derivatives", py_length_derivatives,            METH_VARARGS, NULL},
    {"posterior",          py_posterior,                     METH_VARARGS, NULL},
    {"load",               py_load,                          METH_O, NULL},
    {"save",               py_save,                          METH_VARARGS, NULL},
    {"config",             py_config,                        METH_O, NULL},
    {"n_observed",         py_n_observed,                    METH_O, NULL},
    {"n_skipped",          py_n_skipped,                     METH_O, NULL},
    {"total_weight",       py_total_weight,                  METH_O, NULL},
    {"n_states",           py_n_states,                      METH_O, NULL},
    {"alphabet_info",      py_alphabet_info,                 METH_O, NULL},
    {"alphabets",          py_alphabets,                 METH_NOARGS, NULL},
    {"n_cells",            py_n_cells,                       METH_O, NULL},
    {"n_nonzero_cells",    py_n_nonzero_cells,               METH_O, NULL},
    {"memory_bytes",       py_memory_bytes,                  METH_O, NULL},
    {"live_cache_bytes",   py_live_cache_bytes,              METH_O, NULL},
    {"drop_caches",        py_drop_caches,                   METH_O, NULL},
    {"compact",            py_compact,                       METH_O, NULL},
    {"log_pseq",           py_log_pseq,                      METH_VARARGS, NULL},
    {"log_pseq_batch",     py_log_pseq_batch,                METH_VARARGS, NULL},
    {"explain",            py_explain,                       METH_VARARGS, NULL},
    {"log_partition",      py_log_partition,                 METH_VARARGS, NULL},
    {"log_length_prior",   py_log_length_prior,              METH_VARARGS, NULL},
    {"path_count",         py_path_count,                    METH_VARARGS, NULL},
    {"path_count_total",   py_path_count_total,              METH_O, NULL},
    {"entropy",            py_entropy,                       METH_O, NULL},
    {"hill_number",        py_hill_number,                   METH_VARARGS, NULL},
    {"accumulator_max",    py_accumulator_max,               METH_O, NULL},
    {"spectrum",           py_spectrum,                      METH_VARARGS, NULL},
    {"sample",             py_sample,                        METH_VARARGS, NULL},
    {"subtract",           py_subtract,                      METH_VARARGS, NULL},
    {"add",                py_add,                           METH_VARARGS, NULL},
    {"log_mellin",         py_log_mellin,                    METH_VARARGS, NULL},
    {"log_mellin_many",    py_log_mellin_many,               METH_VARARGS, NULL},
    {"mixture_check_shape", py_mixture_check_shape,            METH_O, NULL},
    {"mixture_inner_product", py_mixture_inner_product,        METH_VARARGS, NULL},
    {"mixture_gram",        py_mixture_gram,                  METH_VARARGS, NULL},
    {"mixture_support_union", py_mixture_support_union,        METH_VARARGS, NULL},
    {"surprisal_moments",  py_surprisal_moments,             METH_VARARGS, NULL},
    {"surprisal_cumulants", py_surprisal_cumulants,          METH_VARARGS, NULL},
    {"surprisal_range",    py_surprisal_range,               METH_O, NULL},
    {"length_marginals",   py_length_marginals,              METH_O, NULL},
    {"top_k",              py_top_k,                         METH_VARARGS, NULL},
    {"surprisal_tilt",     py_surprisal_tilt,                METH_VARARGS, NULL},
    {"surprisal_by_depth", py_surprisal_by_depth,            METH_VARARGS, NULL},
    {"topology",           py_topology,                      METH_VARARGS, NULL},
    {"positional_marginals", py_positional_marginals,        METH_VARARGS, NULL},
    {"positional_banded",    py_positional_banded,           METH_VARARGS, NULL},
    {"insights_logos",       py_insights_logos,              METH_VARARGS, NULL},
    {"insights_logo_cost",   py_insights_logo_cost,          METH_VARARGS, NULL},
    {"expected_counts",    py_expected_counts,               METH_VARARGS, NULL},
    {"query_probability",  py_query_probability,             METH_VARARGS, NULL},
    {"query_marginals",    py_query_marginals,               METH_VARARGS, NULL},
    {"query_entropy",      py_query_entropy,                 METH_VARARGS, NULL},
    {"query_expect",       py_query_expect,                  METH_VARARGS, NULL},
    {"query_position_entropy", py_query_position_entropy,    METH_VARARGS, NULL},
    {"query_pair_joint",   py_query_pair_joint,              METH_VARARGS, NULL},
    {"query_information_matrix", py_query_information_matrix, METH_VARARGS, NULL},
    {"query_cost",         py_query_cost,                    METH_VARARGS, NULL},
    {"query_neighbours",   py_query_neighbours,              METH_VARARGS, NULL},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef beam_module = {
    PyModuleDef_HEAD_INIT, "_cbeam",
    "BEAM: dual-anchored expert model over CDR3 sequences.",
    -1, module_methods, NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC PyInit__cbeam(void) {
    return PyModule_Create(&beam_module);
}
