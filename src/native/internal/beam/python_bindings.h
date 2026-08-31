/**
 * @file python_bindings.h
 * @brief Declarations shared between the binding translation units.
 *
 * The bindings are split by domain, but there is exactly one module
 * initialiser and one method table, both in module.c. Everything module.c
 * names is declared here.
 *
 * Signatures in this file were extracted from the definitions rather than
 * retyped. A binding whose declaration disagrees with its definition compiles
 * on many platforms and then corrupts the stack at call time.
 */
#ifndef BEAM_INTERNAL_PYTHON_BINDINGS_H
#define BEAM_INTERNAL_PYTHON_BINDINGS_H

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "beam/model.h"
#include "beam/publicness.h"

extern const char *BEAM_CAPSULE_NAME;

void capsule_destructor(PyObject *capsule);
double *edge_borrow_abundances(PyObject *abund, Py_ssize_t n, bool *out_ok);
char * *edge_borrow_strings(PyObject *seq, Py_ssize_t *out_n);
PyObject *edge_doubles_to_bytes(const double *v, size_t n);
BeamModel *edge_from_capsule(PyObject *capsule);
PyObject *edge_moment_list(const BeamModel *m, double q, unsigned int order, int cumulants);
int edge_parse_config(PyObject *args, BeamModelConfig *cfg, PyObject **seqs, PyObject **abund);
Py_ssize_t f64_buffer_count(const Py_buffer *view);
int get_f64_buffer(PyObject *obj, Py_buffer *view, int writable, const char *what);
int get_f64_buffers(PyObject **objs, Py_buffer *views, const int *writable, const char **names, int n);
int get_u32_buffer(PyObject *obj, Py_buffer *view, const char *what);
PyObject *py_accumulator_max(PyObject *self, PyObject *arg);
PyObject *py_add(PyObject *self, PyObject *args);
PyObject *py_build(PyObject *self, PyObject *args);
PyObject *py_compact(PyObject *self, PyObject *arg);
PyObject *py_config(PyObject *self, PyObject *arg);
PyObject *py_effective_threads(PyObject *self, PyObject *arg);
PyObject *py_entropy(PyObject *self, PyObject *arg);
PyObject *py_explain(PyObject *self, PyObject *args);
PyObject *py_finalize(PyObject *self, PyObject *arg);
PyObject *py_hill_number(PyObject *self, PyObject *args);
PyObject *py_is_finalized(PyObject *self, PyObject *arg);
PyObject *py_length_derivatives(PyObject *self, PyObject *args);
PyObject *py_length_marginals(PyObject *self, PyObject *arg);
PyObject *py_load(PyObject *self, PyObject *arg);
PyObject *py_log_length_prior(PyObject *self, PyObject *arg);
PyObject *py_log_mellin(PyObject *self, PyObject *args);
PyObject *py_log_partition(PyObject *self, PyObject *arg);
PyObject *py_log_pseq(PyObject *self, PyObject *args);
PyObject *py_log_pseq_batch(PyObject *self, PyObject *args);
PyObject *py_prism_factor(PyObject *self, PyObject *args);
PyObject *py_prism_explain(PyObject *self, PyObject *args);
PyObject *py_memory_budget(PyObject *self, PyObject *arg);
PyObject *py_memory_bytes(PyObject *self, PyObject *arg);
PyObject *py_n_cells(PyObject *self, PyObject *arg);
PyObject *py_n_nonzero_cells(PyObject *self, PyObject *arg);
PyObject *py_n_observed(PyObject *self, PyObject *arg);
PyObject *py_n_skipped(PyObject *self, PyObject *arg);
PyObject *py_n_states(PyObject *self, PyObject *arg);
PyObject *py_new(PyObject *self, PyObject *args);
PyObject *py_observe(PyObject *self, PyObject *args);
PyObject *py_path_count(PyObject *self, PyObject *args);
PyObject *py_path_count_total(PyObject *self, PyObject *arg);
PyObject *py_posterior(PyObject *self, PyObject *args);
PyObject *py_publicness_accumulate(PyObject *self, PyObject *args);
PyObject *py_publicness_accumulate_grouped(PyObject *self, PyObject *args);
PyObject *py_publicness_moments(PyObject *self, PyObject *args);
PyObject *py_publicness_pgf(PyObject *self, PyObject *args);
PyObject *py_sample(PyObject *self, PyObject *args);
PyObject *py_save(PyObject *self, PyObject *args);
PyObject *py_set_memory_budget(PyObject *self, PyObject *arg);
PyObject *py_set_threads(PyObject *self, PyObject *arg);
PyObject *py_spectra_by_length(PyObject *self, PyObject *args);
PyObject *py_spectrum(PyObject *self, PyObject *args);
PyObject *py_spectrum_pair(PyObject *self, PyObject *args);
PyObject *py_subtract(PyObject *self, PyObject *args);
PyObject *py_surprisal_by_depth(PyObject *self, PyObject *args);
PyObject *py_surprisal_cumulants(PyObject *self, PyObject *args);
PyObject *py_surprisal_moments(PyObject *self, PyObject *args);
PyObject *py_surprisal_range(PyObject *self, PyObject *arg);
PyObject *py_surprisal_tilt(PyObject *self, PyObject *args);
PyObject *py_threads(PyObject *self, PyObject *arg);
PyObject *py_top_k(PyObject *self, PyObject *args);
PyObject *py_topology(PyObject *self, PyObject *args);
PyObject *py_total_weight(PyObject *self, PyObject *arg);
void release_f64_buffers(Py_buffer *views, int n);
PyObject *set_beam_error(BeamError err);

#endif /* BEAM_INTERNAL_PYTHON_BINDINGS_H */
