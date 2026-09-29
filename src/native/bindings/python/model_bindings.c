/**
 * @file model_bindings.c
 * @brief Lifecycle, training, persistence, algebra, introspection and runtime.
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
#include "beam/alphabet.h"
#include "beam/publicness.h"

/* Capsule tag. Carried over verbatim so a model pickled or passed between the
 * two packages during the transition is still recognised; it is renamed with
 * everything else in the rename step. */
#include "beam/python_bindings.h"
#include "guards.h"

PyObject *py_new(PyObject *self, PyObject *args) {
    (void)self;
    BeamModelConfig cfg;
    if (!edge_parse_config(args, &cfg, NULL, NULL)) return NULL;
    BeamModel *m = beam_model_new(&cfg);
    /* set_beam_error carries the message AND the right exception type: a
     * model too large for the budget is a MemoryError, not a ValueError, and
     * a caller retrying at a smaller order needs to be able to tell those
     * apart. Hard-coding ValueError here made every refusal look like a bad
     * argument and every refusal message unreachable. */
    if (!m) return set_beam_error(beam_last_error());
    return PyCapsule_New(m, BEAM_CAPSULE_NAME, capsule_destructor);
}
PyObject *py_observe(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap, *seqs, *abund;
    if (!PyArg_ParseTuple(args, "OOO", &cap, &seqs, &abund)) return NULL;
    BeamModel *m = edge_from_capsule(cap);
    if (!m) return NULL;

    Py_ssize_t n = 0;
    const char **v = edge_borrow_strings(seqs, &n);
    if (!v) return NULL;
    bool ok = true;
    double *w = edge_borrow_abundances(abund, n, &ok);
    if (!ok) { PyMem_Free(v); return NULL; }

    BeamError err;
    Py_BEGIN_ALLOW_THREADS
    err = beam_model_observe(m, v, w, (uint32_t)n);
    Py_END_ALLOW_THREADS
    PyMem_Free(v); PyMem_Free(w);
    if (err != BEAM_OK) return set_beam_error(err);
    Py_RETURN_NONE;
}
PyObject *py_drop_caches(PyObject *self, PyObject *arg) {
    (void)self;
    BeamModel *m = edge_from_capsule(arg);
    if (!m) return NULL;
    beam_model_drop_caches(m);
    Py_RETURN_NONE;
}
PyObject *py_finalize(PyObject *self, PyObject *arg) {
    (void)self;
    BeamModel *m = edge_from_capsule(arg);
    if (!m) return NULL;
    BeamError err;
    Py_BEGIN_ALLOW_THREADS
    err = beam_model_finalize(m);
    Py_END_ALLOW_THREADS
    if (err != BEAM_OK) return set_beam_error(err);
    Py_RETURN_NONE;
}
PyObject *py_is_finalized(PyObject *self, PyObject *arg) {
    (void)self;
    BeamModel *m = edge_from_capsule(arg);
    if (!m) return NULL;
    return PyBool_FromLong(beam_model_is_finalized(m) ? 1 : 0);
}
PyObject *py_build(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *seqs, *abund;
    BeamModelConfig cfg;
    if (!edge_parse_config(args, &cfg, &seqs, &abund)) return NULL;

    Py_ssize_t n = 0;
    const char **v = edge_borrow_strings(seqs, &n);
    if (!v) return NULL;
    bool abund_ok = true;
    double *w = edge_borrow_abundances(abund, n, &abund_ok);
    if (!abund_ok) { PyMem_Free(v); return NULL; }

    BeamModel *m = beam_model_new(&cfg);
    if (!m) {
        PyMem_Free(v); PyMem_Free(w);
        return set_beam_error(beam_last_error());
    }

    BeamError err;
    Py_BEGIN_ALLOW_THREADS
    err = beam_model_observe(m, v, w, (uint32_t)n);
    if (err == BEAM_OK) err = beam_model_finalize(m);
    Py_END_ALLOW_THREADS
    PyMem_Free(v); PyMem_Free(w);
    if (err != BEAM_OK) { beam_model_destroy(m); return set_beam_error(err); }
    return PyCapsule_New(m, BEAM_CAPSULE_NAME, capsule_destructor);
}
PyObject *py_load(PyObject *self, PyObject *arg) {
    (void)self;
    const char *path = PyUnicode_AsUTF8(arg);
    if (!path) return NULL;
    BeamModel *m;
    Py_BEGIN_ALLOW_THREADS
    m = beam_model_load(path);
    Py_END_ALLOW_THREADS
    if (!m) {
        if (beam_last_error() != BEAM_OK) {
            /*
             * `beam_model_load` now names its own failure, and so do
             * `beam_model_new` and `beam_model_finalize` when the file is
             * fine but the model cannot be built -- a memory-budget refusal
             * being the case that matters. This branch used to overwrite all
             * of them with one sentence about corruption, which told the
             * reader to rebuild a file that was never damaged.
             */
            return set_beam_error(beam_last_error());
        } else {
            PyErr_Format(PyExc_OSError,
                "could not load a BEAM model from %s (not a model file, "
                "truncated, or corrupt)", path);
        }
        return NULL;
    }
    return PyCapsule_New(m, BEAM_CAPSULE_NAME, capsule_destructor);
}
PyObject *py_save(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap; const char *path;
    if (!PyArg_ParseTuple(args, "Os", &cap, &path)) return NULL;
    BeamModel *m = edge_from_capsule(cap);
    if (!m) return NULL;
    BeamError err;
    Py_BEGIN_ALLOW_THREADS
    err = beam_model_save(m, path);
    Py_END_ALLOW_THREADS
    if (err != BEAM_OK) return set_beam_error(err);
    Py_RETURN_NONE;
}
PyObject *py_config(PyObject *self, PyObject *arg) {
    (void)self;
    BeamModel *m = edge_from_capsule(arg);
    if (!m) return NULL;
    const BeamModelConfig *c = beam_model_config(m);
    PyObject *alpha = PyTuple_New((Py_ssize_t)c->order + 1);
    if (!alpha) return NULL;
    for (uint32_t k = 0; k <= c->order; ++k) {
        PyObject *v = PyFloat_FromDouble(c->alpha[k]);
        if (!v) { Py_DECREF(alpha); return NULL; }
        PyTuple_SET_ITEM(alpha, (Py_ssize_t)k, v);
    }
    PyObject *out = Py_BuildValue(
        "{s:s,s:i,s:i,s:(dd),s:O,s:d,s:O,s:O,s:O}",
        "alphabet", beam_alphabet_info(c->alphabet)->name,
        "order", (int)c->order,
        "max_length", (int)c->max_length,
        "weights", c->weight_forward, c->weight_backward,
        "alpha", alpha,
        "alpha_length", c->alpha_length,
        "alpha_auto", c->alpha_auto ? Py_True : Py_False,
        "alpha_length_auto", c->alpha_length_auto ? Py_True : Py_False,
        "sparse", c->sparse ? Py_True : Py_False);
    Py_DECREF(alpha);
    return out;
}
/*
 * The alphabet descriptor, verbatim.
 *
 * Python needs the per-alphabet defaults to resolve `order=None` and friends,
 * and config.c records a real bug from the last time the two layers each kept
 * their own copy of a default: they disagreed, silently, and both produced a
 * model. Serving them from the descriptor means there is one place to change
 * and nothing to keep in step.
 */
PyObject *py_alphabet_info(PyObject *self, PyObject *arg) {
    (void)self;
    const char *name = PyUnicode_AsUTF8(arg);
    if (!name) return NULL;
    const BeamAlphabetInfo *a = beam_alphabet_by_name(name);
    if (!a) {
        PyErr_Format(PyExc_ValueError, "unknown alphabet %R", arg);
        return NULL;
    }
    return Py_BuildValue(
        "{s:s,s:i,s:i,s:i,s:i,s:i,s:i,s:s#}",
        "name", a->name,
        "id", (int)a->id,
        "residues", (int)a->residues,
        "radix", (int)a->radix,
        "default_order", (int)a->default_order,
        "max_order", (int)a->max_order,
        "default_max_length", (int)a->default_max_length,
        "symbols", a->idx_to_char, (Py_ssize_t)a->residues);
}

PyObject *py_alphabets(PyObject *self, PyObject *arg) {
    (void)self; (void)arg;
    PyObject *t = PyTuple_New(BEAM_ALPHABET_COUNT);
    if (!t) return NULL;
    for (unsigned i = 0; i < BEAM_ALPHABET_COUNT; ++i) {
        PyObject *n = PyUnicode_FromString(
            beam_alphabet_info((BeamAlphabet)i)->name);
        if (!n) { Py_DECREF(t); return NULL; }
        PyTuple_SET_ITEM(t, (Py_ssize_t)i, n);
    }
    return t;
}

PyObject *py_compact(PyObject *self, PyObject *arg) {
    (void)self;
    BeamModel *m = edge_from_capsule(arg);
    if (!m) return NULL;
    BeamError err = beam_model_compact(m);
    if (err != BEAM_OK) return set_beam_error(err);
    Py_RETURN_NONE;
}
PyObject *py_temper(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap; double t;
    if (!PyArg_ParseTuple(args, "Od", &cap, &t)) return NULL;
    BeamModel *m = edge_from_capsule(cap);
    if (!m) return NULL;
    BeamError err;
    Py_BEGIN_ALLOW_THREADS
    err = beam_model_temper(m, t);
    Py_END_ALLOW_THREADS
    if (err != BEAM_OK) return set_beam_error(err);
    Py_RETURN_NONE;
}
PyObject *py_posterior(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap, *prior_cap; double kappa;
    if (!PyArg_ParseTuple(args, "OOd", &cap, &prior_cap, &kappa)) return NULL;
    BeamModel *m = edge_from_capsule(cap);
    BeamModel *p = edge_from_capsule(prior_cap);
    if (!m || !p) return NULL;
    BeamError err;
    Py_BEGIN_ALLOW_THREADS
    err = beam_model_posterior(m, p, kappa);
    if (err == BEAM_OK) err = beam_model_finalize(m);
    Py_END_ALLOW_THREADS
    if (err != BEAM_OK) return set_beam_error(err);
    Py_RETURN_NONE;
}
PyObject *py_set_prior(PyObject *self, PyObject *args) {
    (void)self;
    /* None for either strength hands it to the data. */
    PyObject *cap, *alpha_obj, *alpha_length_obj;
    if (!PyArg_ParseTuple(args, "OOO", &cap, &alpha_obj, &alpha_length_obj))
        return NULL;
    BeamModel *m = edge_from_capsule(cap);
    if (!m) return NULL;
    double alpha_length = NAN;
    if (alpha_length_obj != Py_None) {
        alpha_length = PyFloat_AsDouble(alpha_length_obj);
        if (PyErr_Occurred()) return NULL;
    }
    double alpha[BEAM_MAX_ORDER + 1];
    Py_ssize_t n = 0;
    if (alpha_obj != Py_None) {
        PyObject *fast = PySequence_Fast(alpha_obj, "alpha must be a sequence");
        if (!fast) return NULL;
        n = PySequence_Fast_GET_SIZE(fast);
        if (n < 1 || n > (Py_ssize_t)(BEAM_MAX_ORDER + 1)) {
            Py_DECREF(fast);
            PyErr_Format(PyExc_ValueError, "alpha has %zd entries", n);
            return NULL;
        }
        for (Py_ssize_t k = 0; k < n; ++k) {
            alpha[k] = PyFloat_AsDouble(PySequence_Fast_GET_ITEM(fast, k));
            if (PyErr_Occurred()) { Py_DECREF(fast); return NULL; }
        }
        Py_DECREF(fast);
    }
    BeamError err;
    Py_BEGIN_ALLOW_THREADS
    err = beam_model_set_prior(m, alpha_obj == Py_None ? NULL : alpha,
                               (uint32_t)n, alpha_length);
    Py_END_ALLOW_THREADS
    if (err != BEAM_OK) return set_beam_error(err);
    Py_RETURN_NONE;
}
PyObject *py_subtract(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap, *other_cap;
    if (!PyArg_ParseTuple(args, "OO", &cap, &other_cap)) return NULL;
    BeamModel *m = edge_from_capsule(cap);
    BeamModel *o = edge_from_capsule(other_cap);
    if (!m || !o) return NULL;
    BeamError err;
    Py_BEGIN_ALLOW_THREADS
    err = beam_model_subtract(m, o);
    if (err == BEAM_OK) err = beam_model_finalize(m);
    Py_END_ALLOW_THREADS
    if (err != BEAM_OK) return set_beam_error(err);
    Py_RETURN_NONE;
}
PyObject *py_add(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap, *other_cap;
    if (!PyArg_ParseTuple(args, "OO", &cap, &other_cap)) return NULL;
    BeamModel *m = edge_from_capsule(cap);
    BeamModel *o = edge_from_capsule(other_cap);
    if (!m || !o) return NULL;
    BeamError err;
    Py_BEGIN_ALLOW_THREADS
    err = beam_model_add(m, o);
    if (err == BEAM_OK) err = beam_model_finalize(m);
    Py_END_ALLOW_THREADS
    if (err != BEAM_OK) return set_beam_error(err);
    Py_RETURN_NONE;
}
PyObject *py_set_threads(PyObject *self, PyObject *arg) {
    (void)self;
    long n = PyLong_AsLong(arg);
    if (n == -1 && PyErr_Occurred()) return NULL;
    if (n > INT_MAX) n = INT_MAX;
    beam_model_set_threads((int)n);
    Py_RETURN_NONE;
}
PyObject *py_threads(PyObject *self, PyObject *arg) {
    (void)self; (void)arg;
    return PyLong_FromLong(beam_model_threads());
}
PyObject *py_effective_threads(PyObject *self, PyObject *arg) {
    (void)self; (void)arg;
    return PyLong_FromLong(beam_model_effective_threads());
}
PyObject *py_set_memory_budget(PyObject *self, PyObject *arg) {
    (void)self;
    unsigned long long n = PyLong_AsUnsignedLongLong(arg);
    if (n == (unsigned long long)-1 && PyErr_Occurred()) return NULL;
    beam_model_set_memory_budget((uint64_t)n);
    Py_RETURN_NONE;
}
PyObject *py_memory_budget(PyObject *self, PyObject *arg) {
    (void)self; (void)arg;
    return PyLong_FromUnsignedLongLong(
        (unsigned long long)beam_model_memory_budget());
}
#define BEAM_SCALAR(name, expr, build)                                   \
    PyObject *py_##name(PyObject *self, PyObject *arg) {     \
        (void)self;                                                      \
        BeamModel *m = edge_from_capsule(arg);                        \
        if (!m) return NULL;                                             \
        return build(expr);                                              \
    }

BEAM_SCALAR(n_observed,      beam_model_n_observed(m),      PyLong_FromUnsignedLongLong)
BEAM_SCALAR(n_skipped,       beam_model_n_skipped(m),       PyLong_FromUnsignedLongLong)
BEAM_SCALAR(total_weight,    beam_model_total_weight(m),    PyFloat_FromDouble)
BEAM_SCALAR(n_states,        beam_model_n_states(m),        PyLong_FromUnsignedLong)
BEAM_SCALAR(n_cells,         beam_model_n_cells(m),         PyLong_FromUnsignedLongLong)
BEAM_SCALAR(n_nonzero_cells, beam_model_n_nonzero_cells(m), PyLong_FromUnsignedLongLong)
BEAM_SCALAR(memory_bytes,    beam_model_memory_bytes(m),    PyLong_FromUnsignedLongLong)
BEAM_SCALAR(live_cache_bytes, beam_model_live_cache_bytes(m), PyLong_FromUnsignedLongLong)

/*
 * Provenance, as one tuple rather than three accessors, because the three are
 * always wanted together and three round trips through a capsule to answer
 * one question is three chances to answer it inconsistently.
 */
PyObject *py_provenance(PyObject *self, PyObject *arg) {
    (void)self;
    BeamModel *m = edge_from_capsule(arg);
    if (!m) return NULL;
    int64_t created = 0, obs = 0, fin = 0;
    const BeamError e = beam_model_provenance(m, &created, &obs, &fin);
    if (e != BEAM_OK) return set_beam_error(e);
    return Py_BuildValue("(LLL)", (long long)created, (long long)obs,
                         (long long)fin);
}

/* Where a saved model keeps its build timestamps, as (offset, bytes).
 * A property of the FORMAT, not of any model, so it takes no argument: the
 * byte-level baseline needs it before it has a model to ask. */
PyObject *py_file_provenance_span(PyObject *self, PyObject *args) {
    (void)self; (void)args;
    size_t off = 0, len = 0;
    beam_model_file_provenance_span(&off, &len);
    return Py_BuildValue("(KK)", (unsigned long long)off,
                         (unsigned long long)len);
}

/* Occupied rows and cells per (direction, depth), as two flat lists. */
PyObject *py_depth_occupancy(PyObject *self, PyObject *arg) {
    (void)self;
    BeamModel *m = edge_from_capsule(arg);
    if (!m) return NULL;
    const uint32_t W = beam_model_config(m)->max_length;
    const size_t n = (size_t)2u * W;
    uint64_t *rows = (uint64_t *)malloc(n * sizeof(uint64_t));
    uint64_t *cells = (uint64_t *)malloc(n * sizeof(uint64_t));
    if (!rows || !cells) {
        free(rows); free(cells);
        return beam_py_nomem("the depth occupancy buffers",
                             (uint64_t)2u * n * sizeof(uint64_t));
    }
    BeamError err;
    Py_BEGIN_ALLOW_THREADS
    err = beam_model_depth_occupancy(m, rows, cells);
    Py_END_ALLOW_THREADS
    if (err != BEAM_OK) { free(rows); free(cells); return set_beam_error(err); }
    PyObject *lr = PyList_New((Py_ssize_t)n), *lc = PyList_New((Py_ssize_t)n);
    if (!lr || !lc) { Py_XDECREF(lr); Py_XDECREF(lc); free(rows); free(cells);
                      return NULL; }
    for (size_t i = 0; i < n; ++i) {
        PyList_SET_ITEM(lr, (Py_ssize_t)i,
                        PyLong_FromUnsignedLongLong(rows[i]));
        PyList_SET_ITEM(lc, (Py_ssize_t)i,
                        PyLong_FromUnsignedLongLong(cells[i]));
    }
    free(rows); free(cells);
    return Py_BuildValue("(NN)", lr, lc);
}
/*
 * Not BEAM_SCALAR, which is for accessors that cannot fail. This one reports
 * failure as 0.0 with the error set, and 0.0 is also a legitimate total, so
 * without the check a memory-budget refusal reached Python as "this model
 * generates nothing". `beam_model_path_count_total` clears the error on entry
 * by way of the per-length calls it makes, so a stale one cannot trip it.
 */
PyObject *py_path_count_total(PyObject *self, PyObject *arg) {
    (void)self;
    BeamModel *m = edge_from_capsule(arg);
    if (!m) return NULL;
    double v;
    Py_BEGIN_ALLOW_THREADS
    v = beam_model_path_count_total(m);
    Py_END_ALLOW_THREADS
    if (beam_last_error() != BEAM_OK) return set_beam_error(beam_last_error());
    return PyFloat_FromDouble(v);
}
/*
 * Not BEAM_SCALAR, for the same reason path_count_total is not: this one can
 * fail, and it reports failure as 0.0 with the error set. Zero is also what a
 * model that generates nothing returns, so without the check a refused sweep
 * reached Python as a believable answer -- the reader goes to look at their
 * corpus for a problem that is in their memory budget.
 *
 * `beam_model_accumulator_max` clears the error on entry, so a stale one from
 * an earlier call cannot trip this.
 */
PyObject *py_accumulator_max(PyObject *self, PyObject *arg) {
    (void)self;
    BeamModel *m = edge_from_capsule(arg);
    if (!m) return NULL;
    double v;
    Py_BEGIN_ALLOW_THREADS
    v = beam_model_accumulator_max(m);
    Py_END_ALLOW_THREADS
    if (beam_last_error() != BEAM_OK) return set_beam_error(beam_last_error());
    return PyFloat_FromDouble(v);
}
/*
 * Checks the error, which the plain PyFloat_FromDouble form did not. Both
 * wrapped functions return -INFINITY for a real answer -- the worth of a
 * length the model does not generate -- AND for three error conditions, so
 * the value alone cannot say which happened. The facade bounds `length`
 * before calling, so this guard is reached only through raw `_cbeam`, which
 * is exactly where it was missing.
 */
#define BEAM_LEN_FN(name, call)                                          \
    PyObject *py_##name(PyObject *self, PyObject *args) {    \
        (void)self;                                                      \
        PyObject *cap; unsigned int L;                                   \
        if (!PyArg_ParseTuple(args, "OI", &cap, &L)) return NULL;        \
        BeamModel *m = edge_from_capsule(cap);                        \
        if (!m) return NULL;                                             \
        const double v_ = (call);                                        \
        if (beam_last_error() != BEAM_OK)                                \
            return set_beam_error(beam_last_error());                    \
        return PyFloat_FromDouble(v_);                                   \
    }

BEAM_LEN_FN(log_partition,    beam_model_log_partition(m, L))

/* `log Z(L)` split across its layers, as a list of `length` running masses.
 * The caller differences them; the last entry is `log_partition(L)` exactly. */
PyObject *py_log_partition_by_layer(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap; unsigned int L;
    if (!PyArg_ParseTuple(args, "OI", &cap, &L)) return NULL;
    BeamModel *m = edge_from_capsule(cap);
    if (!m) return NULL;
    if (L == 0u) {
        PyErr_SetString(PyExc_ValueError, "length must be at least 1");
        return NULL;
    }
    double *buf = (double *)PyMem_Malloc((size_t)L * sizeof(double));
    if (!buf) return beam_py_nomem("the per-layer partition buffer",
                                   (uint64_t)L * sizeof(double));
    BeamError err;
    Py_BEGIN_ALLOW_THREADS
    err = beam_model_log_partition_by_layer(m, L, buf);
    Py_END_ALLOW_THREADS
    if (err != BEAM_OK) { PyMem_Free(buf); return set_beam_error(err); }
    PyObject *out = PyList_New((Py_ssize_t)L);
    if (!out) { PyMem_Free(buf); return NULL; }
    for (unsigned int i = 0; i < L; ++i)
        PyList_SET_ITEM(out, (Py_ssize_t)i, PyFloat_FromDouble(buf[i]));
    PyMem_Free(buf);
    return out;
}
BEAM_LEN_FN(log_length_prior, beam_model_log_length_prior(m, L))
