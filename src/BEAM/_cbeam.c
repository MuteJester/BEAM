/*
 * BEAM -- Python bindings for the dual-anchored expert model.
 *
 * Extracted from LZGraphs' `_clzgraph.c`, where the BeamModel bindings lived
 * interleaved with the FlashBack and LZGraph ones. Nothing here depends on
 * those; the C surface it wraps is `beam_model_*` plus the shared error,
 * RNG and CRC helpers, which came across with it.
 *
 * Names are still `beam_*` and `py_*` at this point: this is the
 * transfer, and the rename to BEAM is a separate step so that any behavioural
 * difference introduced by the move shows up before the names change under it.
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
static const char *BEAM_CAPSULE_NAME = "BeamModel";

/*
 * Map a C error code onto a Python exception.
 *
 * Adapted from LZGraphs' version. The gene-data, graph-operation and
 * convergence branches referred to exception types that only exist in that
 * package and to failures this model cannot produce, so they fold into the
 * generic cases rather than being carried across as dead code. The message
 * itself always comes from `beam_error_message()`, which is where the useful
 * detail lives -- a bare code reaches Python as "invalid argument" and is
 * worth nothing to a caller.
 */
static PyObject *set_beam_error(BeamError err) {
    const char *msg = beam_error_message();
    const int has_msg = msg && msg[0];

    switch (err) {
        case BEAM_ERR_ALLOC:
            return PyErr_NoMemory();

        case BEAM_ERR_NULL_ARG:
        case BEAM_ERR_EMPTY_INPUT:
        case BEAM_ERR_INVALID_SEQUENCE:
        case BEAM_ERR_INVALID_VARIANT:
        case BEAM_ERR_LENGTH_MISMATCH:
        case BEAM_ERR_PARAM_OUT_OF_RANGE:
            PyErr_SetString(PyExc_ValueError, has_msg ? msg : "invalid argument");
            return NULL;

        case BEAM_ERR_NOT_BUILT:
        case BEAM_ERR_NO_LIVE_PATHS:
        case BEAM_ERR_HAS_CYCLES:
            PyErr_SetString(PyExc_RuntimeError, has_msg ? msg : "model state error");
            return NULL;

        case BEAM_ERR_IO_OPEN:
        case BEAM_ERR_IO_READ:
        case BEAM_ERR_IO_WRITE:
            PyErr_SetString(PyExc_OSError, has_msg ? msg : "I/O error");
            return NULL;

        case BEAM_ERR_IO_CORRUPT:
        case BEAM_ERR_IO_VERSION:
            PyErr_SetString(PyExc_ValueError,
                            has_msg ? msg : "corrupt or unsupported model file");
            return NULL;

        default:
            PyErr_SetString(PyExc_RuntimeError, has_msg ? msg : "unknown error");
            return NULL;
    }
}

static void capsule_destructor(PyObject *capsule) {
    BeamModel *m = (BeamModel *)PyCapsule_GetPointer(capsule, BEAM_CAPSULE_NAME);
    if (m) beam_model_destroy(m);
}

static BeamModel *edge_from_capsule(PyObject *capsule) {
    return (BeamModel *)PyCapsule_GetPointer(capsule, BEAM_CAPSULE_NAME);
}

static const char **edge_borrow_strings(PyObject *seq, Py_ssize_t *out_n) {
    if (!PyList_Check(seq)) {
        PyErr_SetString(PyExc_TypeError, "expected a list of strings");
        return NULL;
    }
    Py_ssize_t n = PyList_GET_SIZE(seq);
    const char **v = (const char **)PyMem_Malloc((size_t)(n > 0 ? n : 1) * sizeof(char *));
    if (!v) { PyErr_NoMemory(); return NULL; }
    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject *item = PyList_GET_ITEM(seq, i);
        if (!PyUnicode_Check(item)) {
            PyMem_Free(v);
            PyErr_Format(PyExc_TypeError, "sequence %zd is not a string", i);
            return NULL;
        }
        v[i] = PyUnicode_AsUTF8(item);
        if (!v[i]) { PyMem_Free(v); return NULL; }
    }
    *out_n = n;
    return v;
}

static double *edge_borrow_abundances(PyObject *abund, Py_ssize_t n, bool *out_ok) {
    *out_ok = true;
    if (abund == Py_None) return NULL;
    PyObject *fast = PySequence_Fast(abund, "abundances must be a sequence");
    if (!fast) { *out_ok = false; return NULL; }
    Py_ssize_t an = PySequence_Fast_GET_SIZE(fast);
    if (an != n) {
        Py_DECREF(fast);
        PyErr_Format(PyExc_ValueError,
                     "abundances has %zd entries but sequences has %zd", an, n);
        *out_ok = false;
        return NULL;
    }
    double *w = (double *)PyMem_Malloc((size_t)(n > 0 ? n : 1) * sizeof(double));
    if (!w) { Py_DECREF(fast); PyErr_NoMemory(); *out_ok = false; return NULL; }
    for (Py_ssize_t i = 0; i < n; i++) {
        w[i] = PyFloat_AsDouble(PySequence_Fast_GET_ITEM(fast, i));
        if (PyErr_Occurred()) {
            PyMem_Free(w); Py_DECREF(fast); *out_ok = false; return NULL;
        }
    }
    Py_DECREF(fast);
    return w;
}

static int edge_parse_config(PyObject *args, BeamModelConfig *cfg,
                             PyObject **seqs, PyObject **abund) {
    int order, depth_cap, max_length, strict, sparse;
    double cw, cm, cl, k0, k1, k2, kw, kl, wf, wb;
    const char *fmt = seqs ? "OOiii(ddd)(ddd)dd(dd)pp" :
                             "iii(ddd)(ddd)dd(dd)pp";
    int ok;
    if (seqs)
        ok = PyArg_ParseTuple(args, fmt, seqs, abund, &order, &depth_cap,
                              &max_length, &cw, &cm, &cl, &k0, &k1, &k2,
                              &kw, &kl, &wf, &wb, &strict, &sparse);
    else
        ok = PyArg_ParseTuple(args, fmt, &order, &depth_cap, &max_length,
                              &cw, &cm, &cl, &k0, &k1, &k2, &kw, &kl, &wf, &wb,
                              &strict, &sparse);
    if (!ok) return 0;
    beam_model_config_default(cfg);
    cfg->order = (uint8_t)order;
    cfg->depth_cap = (uint8_t)depth_cap;
    cfg->max_length = (uint8_t)max_length;
    cfg->coef_wide = cw; cfg->coef_mid = cm; cfg->coef_len = cl;
    cfg->kappa[0] = k0; cfg->kappa[1] = k1; cfg->kappa[2] = k2;
    cfg->kappa_wide = kw; cfg->kappa_len = kl;
    cfg->weight_forward = wf; cfg->weight_backward = wb;
    cfg->strict = (bool)strict;
    cfg->sparse = (bool)sparse;
    return 1;
}

static PyObject *py_new(PyObject *self, PyObject *args) {
    (void)self;
    BeamModelConfig cfg;
    if (!edge_parse_config(args, &cfg, NULL, NULL)) return NULL;
    BeamModel *m = beam_model_new(&cfg);
    if (!m) {
        PyErr_SetString(PyExc_ValueError, "invalid BeamModel configuration");
        return NULL;
    }
    return PyCapsule_New(m, BEAM_CAPSULE_NAME, capsule_destructor);
}

static PyObject *py_observe(PyObject *self, PyObject *args) {
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

static PyObject *py_finalize(PyObject *self, PyObject *arg) {
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

static PyObject *py_is_finalized(PyObject *self, PyObject *arg) {
    (void)self;
    BeamModel *m = edge_from_capsule(arg);
    if (!m) return NULL;
    return PyBool_FromLong(beam_model_is_finalized(m) ? 1 : 0);
}

static PyObject *py_set_threads(PyObject *self, PyObject *arg) {
    (void)self;
    long n = PyLong_AsLong(arg);
    if (n == -1 && PyErr_Occurred()) return NULL;
    if (n > INT_MAX) n = INT_MAX;
    beam_model_set_threads((int)n);
    Py_RETURN_NONE;
}

static PyObject *py_threads(PyObject *self, PyObject *arg) {
    (void)self; (void)arg;
    return PyLong_FromLong(beam_model_threads());
}

static PyObject *py_effective_threads(PyObject *self, PyObject *arg) {
    (void)self; (void)arg;
    return PyLong_FromLong(beam_model_effective_threads());
}

static PyObject *py_set_memory_budget(PyObject *self, PyObject *arg) {
    (void)self;
    unsigned long long n = PyLong_AsUnsignedLongLong(arg);
    if (n == (unsigned long long)-1 && PyErr_Occurred()) return NULL;
    beam_model_set_memory_budget((uint64_t)n);
    Py_RETURN_NONE;
}

static PyObject *py_memory_budget(PyObject *self, PyObject *arg) {
    (void)self; (void)arg;
    return PyLong_FromUnsignedLongLong(
        (unsigned long long)beam_model_memory_budget());
}

static PyObject *edge_doubles_to_bytes(const double *v, size_t n) {
    return PyBytes_FromStringAndSize((const char *)v,
                                     (Py_ssize_t)(n * sizeof(double)));
}

static PyObject *py_spectrum_pair(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap; unsigned int length, bins; double spacing;
    if (!PyArg_ParseTuple(args, "OIId", &cap, &length, &bins, &spacing))
        return NULL;
    BeamModel *m = edge_from_capsule(cap);
    if (!m) return NULL;
    if (bins < 2) {
        PyErr_SetString(PyExc_ValueError, "bins must be at least 2");
        return NULL;
    }
    double *cw = (double *)PyMem_Malloc((size_t)bins * sizeof(double));
    double *gw = (double *)PyMem_Malloc((size_t)bins * sizeof(double));
    if (!cw || !gw) { PyMem_Free(cw); PyMem_Free(gw); return PyErr_NoMemory(); }
    double ds = 0.0;
    BeamError err;
    Py_BEGIN_ALLOW_THREADS
    err = beam_model_spectrum_pair(m, length, bins, spacing, cw, gw, &ds);
    Py_END_ALLOW_THREADS
    if (err != BEAM_OK) { PyMem_Free(cw); PyMem_Free(gw); return set_beam_error(err); }

    PyObject *cb = edge_doubles_to_bytes(cw, bins);
    PyObject *gb = edge_doubles_to_bytes(gw, bins);
    PyMem_Free(cw); PyMem_Free(gw);
    if (!cb || !gb) { Py_XDECREF(cb); Py_XDECREF(gb); return NULL; }
    return Py_BuildValue("(NNd)", cb, gb, ds);
}

static PyObject *py_spectra_by_length(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap; int counting; unsigned int max_length, bins; double spacing;
    if (!PyArg_ParseTuple(args, "OpIId", &cap, &counting, &max_length, &bins,
                          &spacing))
        return NULL;
    BeamModel *m = edge_from_capsule(cap);
    if (!m) return NULL;
    if (bins < 2) {
        PyErr_SetString(PyExc_ValueError, "bins must be at least 2");
        return NULL;
    }
    unsigned int rows = max_length ? max_length + 1u
                                   : (unsigned int)beam_model_config(m)->max_length + 1u;
    size_t n = (size_t)rows * (size_t)bins;
    double *w = (double *)PyMem_Calloc(n ? n : 1, sizeof(double));
    if (!w) return PyErr_NoMemory();
    double ds = 0.0;
    BeamError err;
    Py_BEGIN_ALLOW_THREADS
    err = beam_model_spectra_by_length(m, counting ? true : false,
                                           max_length, bins, spacing, w, &ds);
    Py_END_ALLOW_THREADS
    if (err != BEAM_OK) { PyMem_Free(w); return set_beam_error(err); }
    PyObject *b = edge_doubles_to_bytes(w, n);
    PyMem_Free(w);
    if (!b) return NULL;
    return Py_BuildValue("(NIId)", b, rows, bins, ds);
}

static PyObject *py_length_derivatives(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap; double q; unsigned int order;
    if (!PyArg_ParseTuple(args, "OdI", &cap, &q, &order)) return NULL;
    BeamModel *m = edge_from_capsule(cap);
    if (!m) return NULL;
    unsigned int rows = (unsigned int)beam_model_config(m)->max_length + 1u;
    unsigned int cols = order + 1u;
    size_t n = (size_t)rows * (size_t)cols;
    double *out = (double *)PyMem_Calloc(n, sizeof(double));
    if (!out) return PyErr_NoMemory();
    BeamError err;
    Py_BEGIN_ALLOW_THREADS
    err = beam_model_length_derivatives(m, q, order, out);
    Py_END_ALLOW_THREADS
    if (err != BEAM_OK) { PyMem_Free(out); return set_beam_error(err); }
    PyObject *b = edge_doubles_to_bytes(out, n);
    PyMem_Free(out);
    if (!b) return NULL;
    return Py_BuildValue("(NII)", b, rows, cols);
}

static PyObject *py_posterior(PyObject *self, PyObject *args) {
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

static PyObject *py_build(PyObject *self, PyObject *args) {
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
        PyErr_SetString(PyExc_ValueError, "invalid BeamModel configuration");
        return NULL;
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

static PyObject *py_load(PyObject *self, PyObject *arg) {
    (void)self;
    const char *path = PyUnicode_AsUTF8(arg);
    if (!path) return NULL;
    BeamModel *m;
    Py_BEGIN_ALLOW_THREADS
    m = beam_model_load(path);
    Py_END_ALLOW_THREADS
    if (!m) {
        /* A pre-split EDGE file is a common and confusing way to land here,
         * so name it rather than leaving it to look like corruption. */
        if (beam_model_is_legacy_file(path)) {
            PyErr_Format(PyExc_OSError,
                "%s was written by EDGE, before the split into BEAM. The "
                "format is the same but the file identifier is not, and such "
                "files are not loadable. Rebuild the model with BEAM.", path);
        } else {
            PyErr_Format(PyExc_OSError,
                "could not load a BEAM model from %s (not a model file, "
                "truncated, or corrupt)", path);
        }
        return NULL;
    }
    return PyCapsule_New(m, BEAM_CAPSULE_NAME, capsule_destructor);
}

static PyObject *py_save(PyObject *self, PyObject *args) {
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

static PyObject *py_config(PyObject *self, PyObject *arg) {
    (void)self;
    BeamModel *m = edge_from_capsule(arg);
    if (!m) return NULL;
    const BeamModelConfig *c = beam_model_config(m);
    return Py_BuildValue(
        "{s:i,s:i,s:i,s:(ddd),s:(ddd),s:d,s:d,s:(dd),s:O,s:O}",
        "order", (int)c->order,
        "depth_cap", (int)c->depth_cap,
        "max_length", (int)c->max_length,
        "coefficients", c->coef_wide, c->coef_mid, c->coef_len,
        "kappa", c->kappa[0], c->kappa[1], c->kappa[2],
        "kappa_wide", c->kappa_wide,
        "kappa_len", c->kappa_len,
        "weights", c->weight_forward, c->weight_backward,
        "strict", c->strict ? Py_True : Py_False,
        "sparse", c->sparse ? Py_True : Py_False);
}

#define BEAM_SCALAR(name, expr, build)                                   \
    static PyObject *py_##name(PyObject *self, PyObject *arg) {     \
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
BEAM_SCALAR(path_count_total, beam_model_path_count_total(m), PyFloat_FromDouble)
BEAM_SCALAR(accumulator_max, beam_model_accumulator_max(m), PyFloat_FromDouble)

static PyObject *py_compact(PyObject *self, PyObject *arg) {
    (void)self;
    BeamModel *m = edge_from_capsule(arg);
    if (!m) return NULL;
    BeamError err = beam_model_compact(m);
    if (err != BEAM_OK) return set_beam_error(err);
    Py_RETURN_NONE;
}

static PyObject *py_log_pseq(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap; const char *seq;
    if (!PyArg_ParseTuple(args, "Os", &cap, &seq)) return NULL;
    BeamModel *m = edge_from_capsule(cap);
    if (!m) return NULL;
    return PyFloat_FromDouble(beam_model_log_pseq(m, seq));
}

static PyObject *py_log_pseq_batch(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap, *seqs;
    if (!PyArg_ParseTuple(args, "OO", &cap, &seqs)) return NULL;
    BeamModel *m = edge_from_capsule(cap);
    if (!m) return NULL;

    Py_ssize_t n = 0;
    const char **v = edge_borrow_strings(seqs, &n);
    if (!v) return NULL;
    double *out = (double *)PyMem_Malloc((size_t)(n > 0 ? n : 1) * sizeof(double));
    if (!out) { PyMem_Free(v); return PyErr_NoMemory(); }

    BeamError err;
    Py_BEGIN_ALLOW_THREADS
    err = beam_model_log_pseq_batch(m, v, (uint32_t)n, out);
    Py_END_ALLOW_THREADS
    PyMem_Free(v);
    if (err != BEAM_OK) { PyMem_Free(out); return set_beam_error(err); }

    PyObject *list = PyList_New(n);
    if (!list) { PyMem_Free(out); return NULL; }
    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject *f = PyFloat_FromDouble(out[i]);
        if (!f) { Py_DECREF(list); PyMem_Free(out); return NULL; }
        PyList_SET_ITEM(list, i, f);
    }
    PyMem_Free(out);
    return list;
}

static PyObject *py_explain(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap; const char *seq;
    if (!PyArg_ParseTuple(args, "Os", &cap, &seq)) return NULL;
    BeamModel *m = edge_from_capsule(cap);
    if (!m) return NULL;

    double fwd[BEAM_MAX_LENGTH], bwd[BEAM_MAX_LENGTH];
    uint32_t len = 0;
    BeamError err = beam_model_explain(m, seq, fwd, bwd, &len);
    if (err != BEAM_OK) return set_beam_error(err);

    PyObject *lf = PyList_New(len), *lb = PyList_New(len);
    if (!lf || !lb) { Py_XDECREF(lf); Py_XDECREF(lb); return NULL; }
    for (uint32_t i = 0; i < len; i++) {
        PyList_SET_ITEM(lf, i, PyFloat_FromDouble(fwd[i]));
        PyList_SET_ITEM(lb, i, PyFloat_FromDouble(bwd[i]));
    }
    return Py_BuildValue("(NN)", lf, lb);
}

#define BEAM_LEN_FN(name, call)                                          \
    static PyObject *py_##name(PyObject *self, PyObject *args) {    \
        (void)self;                                                      \
        PyObject *cap; unsigned int L;                                   \
        if (!PyArg_ParseTuple(args, "OI", &cap, &L)) return NULL;        \
        BeamModel *m = edge_from_capsule(cap);                        \
        if (!m) return NULL;                                             \
        return PyFloat_FromDouble(call);                                 \
    }

BEAM_LEN_FN(log_partition,    beam_model_log_partition(m, L))
BEAM_LEN_FN(log_length_prior, beam_model_log_length_prior(m, L))

static PyObject *py_path_count(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap; unsigned int L;
    if (!PyArg_ParseTuple(args, "OI", &cap, &L)) return NULL;
    BeamModel *m = edge_from_capsule(cap);
    if (!m) return NULL;
    double v;
    Py_BEGIN_ALLOW_THREADS
    v = beam_model_path_count(m, L);
    Py_END_ALLOW_THREADS
    return PyFloat_FromDouble(v);
}

static PyObject *py_entropy(PyObject *self, PyObject *arg) {
    (void)self;
    BeamModel *m = edge_from_capsule(arg);
    if (!m) return NULL;
    double h = 0.0;
    BeamError err;
    Py_BEGIN_ALLOW_THREADS
    err = beam_model_entropy(m, &h);
    Py_END_ALLOW_THREADS
    if (err != BEAM_OK) return set_beam_error(err);
    return PyFloat_FromDouble(h);
}

static PyObject *py_hill_number(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap; double alpha;
    if (!PyArg_ParseTuple(args, "Od", &cap, &alpha)) return NULL;
    BeamModel *m = edge_from_capsule(cap);
    if (!m) return NULL;
    double d = 0.0;
    BeamError err;
    Py_BEGIN_ALLOW_THREADS
    err = beam_model_hill_number(m, alpha, &d);
    Py_END_ALLOW_THREADS
    if (err != BEAM_OK) return set_beam_error(err);
    return PyFloat_FromDouble(d);
}

static PyObject *py_spectrum(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap; int counting; unsigned int length, bins; double spacing;
    if (!PyArg_ParseTuple(args, "OpIId", &cap, &counting, &length, &bins, &spacing))
        return NULL;
    BeamModel *m = edge_from_capsule(cap);
    if (!m) return NULL;
    if (bins < 2) {
        PyErr_SetString(PyExc_ValueError, "bins must be at least 2");
        return NULL;
    }
    double *w = (double *)PyMem_Malloc((size_t)bins * sizeof(double));
    if (!w) return PyErr_NoMemory();
    double ds = 0.0;
    BeamError err;
    Py_BEGIN_ALLOW_THREADS
    err = beam_model_spectrum(m, counting ? true : false, length, bins,
                                  spacing, w, &ds);
    Py_END_ALLOW_THREADS
    if (err != BEAM_OK) { PyMem_Free(w); return set_beam_error(err); }

    PyObject *list = PyList_New(bins);
    if (!list) { PyMem_Free(w); return NULL; }
    for (unsigned int i = 0; i < bins; i++)
        PyList_SET_ITEM(list, i, PyFloat_FromDouble(w[i]));
    PyMem_Free(w);
    return Py_BuildValue("(Nd)", list, ds);
}

static PyObject *py_sample(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap; unsigned int n, length; unsigned long long seed;
    if (!PyArg_ParseTuple(args, "OIIK", &cap, &n, &length, &seed)) return NULL;
    BeamModel *m = edge_from_capsule(cap);
    if (!m) return NULL;
    const BeamModelConfig *c = beam_model_config(m);
    const uint32_t stride = (uint32_t)c->max_length + 1u;

    char *buf = (char *)PyMem_Calloc((size_t)n * stride + 1u, 1);
    if (!buf) return PyErr_NoMemory();
    BeamRng rng;
    beam_rng_seed(&rng, (uint64_t)seed);
    BeamError err;
    Py_BEGIN_ALLOW_THREADS
    err = beam_model_sample(m, n, length, &rng, buf, NULL);
    Py_END_ALLOW_THREADS
    if (err != BEAM_OK) { PyMem_Free(buf); return set_beam_error(err); }

    PyObject *list = PyList_New(n);
    if (!list) { PyMem_Free(buf); return NULL; }
    for (unsigned int i = 0; i < n; i++) {
        PyObject *s = PyUnicode_FromString(buf + (size_t)i * stride);
        if (!s) { Py_DECREF(list); PyMem_Free(buf); return NULL; }
        PyList_SET_ITEM(list, i, s);
    }
    PyMem_Free(buf);
    return list;
}

static PyObject *py_subtract(PyObject *self, PyObject *args) {
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

static PyObject *py_add(PyObject *self, PyObject *args) {
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

static PyObject *py_log_mellin(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap; double q;
    if (!PyArg_ParseTuple(args, "Od", &cap, &q)) return NULL;
    BeamModel *m = edge_from_capsule(cap);
    if (!m) return NULL;
    double v = 0.0;
    BeamError err;
    Py_BEGIN_ALLOW_THREADS
    err = beam_model_log_mellin(m, q, &v);
    Py_END_ALLOW_THREADS
    if (err != BEAM_OK) return set_beam_error(err);
    return PyFloat_FromDouble(v);
}

static PyObject *edge_moment_list(const BeamModel *m, double q,
                                  unsigned int order, int cumulants) {
    const unsigned int n = cumulants ? order : order + 1u;
    double *buf = (double *)PyMem_Calloc(n ? n : 1u, sizeof(double));
    if (!buf) return PyErr_NoMemory();
    BeamError err;
    Py_BEGIN_ALLOW_THREADS
    err = cumulants ? beam_model_surprisal_cumulants(m, q, order, buf)
                    : beam_model_surprisal_moments(m, q, order, buf);
    Py_END_ALLOW_THREADS
    if (err != BEAM_OK) { PyMem_Free(buf); return set_beam_error(err); }
    PyObject *list = PyList_New(n);
    if (!list) { PyMem_Free(buf); return NULL; }
    for (unsigned int i = 0; i < n; i++)
        PyList_SET_ITEM(list, i, PyFloat_FromDouble(buf[i]));
    PyMem_Free(buf);
    return list;
}

static PyObject *py_surprisal_moments(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap; double q; unsigned int order;
    if (!PyArg_ParseTuple(args, "OdI", &cap, &q, &order)) return NULL;
    BeamModel *m = edge_from_capsule(cap);
    if (!m) return NULL;
    return edge_moment_list(m, q, order, 0);
}

static PyObject *py_surprisal_cumulants(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap; double q; unsigned int order;
    if (!PyArg_ParseTuple(args, "OdI", &cap, &q, &order)) return NULL;
    BeamModel *m = edge_from_capsule(cap);
    if (!m) return NULL;
    return edge_moment_list(m, q, order, 1);
}

static PyObject *py_surprisal_range(PyObject *self, PyObject *arg) {
    (void)self;
    BeamModel *m = edge_from_capsule(arg);
    if (!m) return NULL;
    double lo = 0.0, hi = 0.0;
    BeamError err;
    Py_BEGIN_ALLOW_THREADS
    err = beam_model_surprisal_range(m, &lo, &hi);
    Py_END_ALLOW_THREADS
    if (err != BEAM_OK) return set_beam_error(err);
    return Py_BuildValue("(dd)", lo, hi);
}

static PyObject *py_length_marginals(PyObject *self, PyObject *arg) {
    (void)self;
    BeamModel *m = edge_from_capsule(arg);
    if (!m) return NULL;
    const unsigned int n = (unsigned int)beam_model_config(m)->max_length + 1u;
    double *c = (double *)PyMem_Calloc(n, sizeof(double));
    double *g = (double *)PyMem_Calloc(n, sizeof(double));
    double *s = (double *)PyMem_Calloc(n, sizeof(double));
    double *v = (double *)PyMem_Calloc(n, sizeof(double));
    if (!c || !g || !s || !v) {
        PyMem_Free(c); PyMem_Free(g); PyMem_Free(s); PyMem_Free(v);
        return PyErr_NoMemory();
    }
    BeamError err;
    Py_BEGIN_ALLOW_THREADS
    err = beam_model_length_marginals(m, c, g, s, v);
    Py_END_ALLOW_THREADS
    if (err != BEAM_OK) {
        PyMem_Free(c); PyMem_Free(g); PyMem_Free(s); PyMem_Free(v);
        return set_beam_error(err);
    }

    PyObject *lc = PyList_New(n), *lg = PyList_New(n);
    PyObject *ls = PyList_New(n), *lv = PyList_New(n);
    if (!lc || !lg || !ls || !lv) {
        Py_XDECREF(lc); Py_XDECREF(lg); Py_XDECREF(ls); Py_XDECREF(lv);
        PyMem_Free(c); PyMem_Free(g); PyMem_Free(s); PyMem_Free(v);
        return NULL;
    }
    for (unsigned int i = 0; i < n; i++) {
        PyList_SET_ITEM(lc, i, PyFloat_FromDouble(c[i]));
        PyList_SET_ITEM(lg, i, PyFloat_FromDouble(g[i]));
        PyList_SET_ITEM(ls, i, PyFloat_FromDouble(s[i]));
        PyList_SET_ITEM(lv, i, PyFloat_FromDouble(v[i]));
    }
    PyMem_Free(c); PyMem_Free(g); PyMem_Free(s); PyMem_Free(v);
    return Py_BuildValue("(NNNN)", lc, lg, ls, lv);
}

static PyObject *py_top_k(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap; unsigned int k, length;
    if (!PyArg_ParseTuple(args, "OII", &cap, &k, &length)) return NULL;
    BeamModel *m = edge_from_capsule(cap);
    if (!m) return NULL;
    const unsigned int stride = (unsigned int)beam_model_config(m)->max_length + 1u;

    char *buf = (char *)PyMem_Calloc((size_t)k * stride + 1u, 1);
    double *lp = (double *)PyMem_Calloc(k ? k : 1u, sizeof(double));
    if (!buf || !lp) { PyMem_Free(buf); PyMem_Free(lp); return PyErr_NoMemory(); }
    uint32_t n = 0;
    BeamError err;
    Py_BEGIN_ALLOW_THREADS
    err = beam_model_top_k(m, k, length, buf, lp, &n);
    Py_END_ALLOW_THREADS
    if (err != BEAM_OK) { PyMem_Free(buf); PyMem_Free(lp); return set_beam_error(err); }

    PyObject *seqs = PyList_New(n), *probs = PyList_New(n);
    if (!seqs || !probs) {
        Py_XDECREF(seqs); Py_XDECREF(probs);
        PyMem_Free(buf); PyMem_Free(lp);
        return NULL;
    }
    for (uint32_t i = 0; i < n; i++) {
        PyObject *str = PyUnicode_FromString(buf + (size_t)i * stride);
        if (!str) { Py_DECREF(seqs); Py_DECREF(probs); PyMem_Free(buf); PyMem_Free(lp); return NULL; }
        PyList_SET_ITEM(seqs, i, str);
        PyList_SET_ITEM(probs, i, PyFloat_FromDouble(lp[i]));
    }
    PyMem_Free(buf); PyMem_Free(lp);
    return Py_BuildValue("(NN)", seqs, probs);
}

static PyObject *py_surprisal_tilt(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap; double q;
    if (!PyArg_ParseTuple(args, "Od", &cap, &q)) return NULL;
    BeamModel *m = edge_from_capsule(cap);
    if (!m) return NULL;
    double lm = 0.0, mean = 0.0, var = 0.0;
    BeamError err;
    Py_BEGIN_ALLOW_THREADS
    err = beam_model_surprisal_tilt(m, q, &lm, &mean, &var);
    Py_END_ALLOW_THREADS
    if (err != BEAM_OK) return set_beam_error(err);
    return Py_BuildValue("(ddd)", lm, mean, var);
}

static PyObject *py_surprisal_by_depth(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap; double q;
    if (!PyArg_ParseTuple(args, "Od", &cap, &q)) return NULL;
    BeamModel *m = edge_from_capsule(cap);
    if (!m) return NULL;
    const unsigned int n = (unsigned int)beam_model_config(m)->max_length;
    double *buf = (double *)PyMem_Calloc(n, sizeof(double));
    if (!buf) return PyErr_NoMemory();
    double konst = 0.0;
    BeamError err;
    Py_BEGIN_ALLOW_THREADS
    err = beam_model_surprisal_by_depth(m, q, buf, &konst);
    Py_END_ALLOW_THREADS
    if (err != BEAM_OK) { PyMem_Free(buf); return set_beam_error(err); }
    PyObject *list = PyList_New(n);
    if (!list) { PyMem_Free(buf); return NULL; }
    for (unsigned int i = 0; i < n; i++)
        PyList_SET_ITEM(list, i, PyFloat_FromDouble(buf[i]));
    PyMem_Free(buf);
    return Py_BuildValue("(Nd)", list, konst);
}

static PyObject *py_topology(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap; unsigned int length;
    if (!PyArg_ParseTuple(args, "OI", &cap, &length)) return NULL;
    BeamModel *m = edge_from_capsule(cap);
    if (!m) return NULL;
    uint32_t *layers = (uint32_t *)PyMem_Calloc(length + 1u, sizeof(uint32_t));
    if (!layers) return PyErr_NoMemory();
    double nodes = 0.0, edges = 0.0;
    BeamError err;
    Py_BEGIN_ALLOW_THREADS
    err = beam_model_topology(m, length, &nodes, &edges, layers);
    Py_END_ALLOW_THREADS
    if (err != BEAM_OK) { PyMem_Free(layers); return set_beam_error(err); }
    PyObject *list = PyList_New(length + 1u);
    if (!list) { PyMem_Free(layers); return NULL; }
    for (unsigned int i = 0; i <= length; i++)
        PyList_SET_ITEM(list, i, PyLong_FromUnsignedLong(layers[i]));
    PyMem_Free(layers);
    return Py_BuildValue("(ddN)", nodes, edges, list);
}

/* ── Buffer helpers for the publicness batch API ───────────── */

/* The publicness entry points move one PMF of up to 65,536 doubles per atom,
 * which is well past the point where a Python list round-trip makes sense.
 * They take the buffer protocol instead. Every array is float64 so the format
 * check stays portable across platforms; a complex128 array is handed over as
 * its float64 view, and integral quantities (multiplicities, bin edges) as
 * exact float64 integers. */

static int get_f64_buffer(PyObject *obj, Py_buffer *view, int writable,
                          const char *what) {
    int flags = PyBUF_C_CONTIGUOUS | PyBUF_FORMAT;
    if (writable) flags |= PyBUF_WRITABLE;

    if (PyObject_GetBuffer(obj, view, flags) != 0) {
        PyErr_Clear();
        PyErr_Format(PyExc_TypeError,
                     "%s must be a C-contiguous%s float64 array",
                     what, writable ? ", writable" : "");
        return 0;
    }
    if (view->itemsize != (Py_ssize_t)sizeof(double) ||
        !view->format || strcmp(view->format, "d") != 0) {
        PyBuffer_Release(view);
        PyErr_Format(PyExc_TypeError, "%s must have dtype float64", what);
        return 0;
    }
    return 1;
}

static void release_f64_buffers(Py_buffer *views, int n) {
    for (int i = 0; i < n; i++) PyBuffer_Release(&views[i]);
}

/* Acquire several float64 buffers at once, releasing what was already taken
 * if any of them fails. */
static int get_f64_buffers(PyObject **objs, Py_buffer *views,
                           const int *writable, const char **names, int n) {
    for (int i = 0; i < n; i++) {
        if (!get_f64_buffer(objs[i], &views[i], writable[i], names[i])) {
            release_f64_buffers(views, i);
            return 0;
        }
    }
    return 1;
}

static Py_ssize_t f64_buffer_count(const Py_buffer *view) {
    return view->len / (Py_ssize_t)sizeof(double);
}

static int get_u32_buffer(PyObject *obj, Py_buffer *view, const char *what) {
    if (PyObject_GetBuffer(obj, view, PyBUF_C_CONTIGUOUS | PyBUF_FORMAT) != 0) {
        PyErr_Clear();
        PyErr_Format(PyExc_TypeError,
                     "%s must be a C-contiguous uint32 array", what);
        return 0;
    }
    if (view->itemsize != (Py_ssize_t)sizeof(uint32_t) ||
        !view->format || strcmp(view->format, "I") != 0) {
        PyBuffer_Release(view);
        PyErr_Format(PyExc_TypeError, "%s must have dtype uint32", what);
        return 0;
    }
    return 1;
}

/* ── publicness_moments(p, depths, mult, mean, var) → None ── */

static PyObject *py_publicness_moments(PyObject *self, PyObject *args) {
    (void)self;
    static const int writable[5] = {0, 0, 0, 1, 1};
    static const char *names[5] = {"probabilities", "depths", "multiplicities",
                                   "mean output", "variance output"};
    PyObject *objs[5];
    Py_buffer views[5];
    Py_ssize_t n_atoms, n_groups;
    BeamError err;

    if (!PyArg_ParseTuple(args, "OOOOO", &objs[0], &objs[1], &objs[2],
                          &objs[3], &objs[4])) return NULL;
    if (!get_f64_buffers(objs, views, writable, names, 5)) return NULL;

    n_atoms = f64_buffer_count(&views[0]);
    n_groups = f64_buffer_count(&views[1]);
    if (f64_buffer_count(&views[2]) != n_groups ||
        f64_buffer_count(&views[3]) != n_atoms ||
        f64_buffer_count(&views[4]) != n_atoms) {
        release_f64_buffers(views, 5);
        PyErr_SetString(PyExc_ValueError,
            "depths and multiplicities must be the same length, and both "
            "outputs must match the probability count");
        return NULL;
    }
    if (n_atoms > UINT32_MAX || n_groups > UINT32_MAX) {
        release_f64_buffers(views, 5);
        PyErr_SetString(PyExc_OverflowError, "batch exceeds uint32 limit");
        return NULL;
    }

    Py_BEGIN_ALLOW_THREADS
    err = beam_publicness_moments((const double *)views[0].buf,
                                 (uint32_t)n_atoms,
                                 (const double *)views[1].buf,
                                 (const double *)views[2].buf,
                                 (uint32_t)n_groups,
                                 (double *)views[3].buf,
                                 (double *)views[4].buf);
    Py_END_ALLOW_THREADS

    release_f64_buffers(views, 5);
    if (err != BEAM_OK) return set_beam_error(err);
    Py_RETURN_NONE;
}

/* ── publicness_pgf(p, depths, mult, out) → None ──────────── */

static PyObject *py_publicness_pgf(PyObject *self, PyObject *args) {
    (void)self;
    static const int writable[4] = {0, 0, 0, 1};
    static const char *names[4] = {"probabilities", "depths", "multiplicities",
                                   "generating-function output"};
    PyObject *objs[4];
    Py_buffer views[4];
    Py_ssize_t n_atoms, n_groups, out_count, k_fft;
    BeamError err;

    if (!PyArg_ParseTuple(args, "OOOO", &objs[0], &objs[1], &objs[2],
                          &objs[3])) return NULL;
    if (!get_f64_buffers(objs, views, writable, names, 4)) return NULL;

    n_atoms = f64_buffer_count(&views[0]);
    n_groups = f64_buffer_count(&views[1]);
    out_count = f64_buffer_count(&views[3]);
    /* The output is the float64 view of an (n_atoms, k_fft) complex array. */
    k_fft = (n_atoms > 0 && out_count % (2 * n_atoms) == 0)
                ? out_count / (2 * n_atoms) : -1;

    if (f64_buffer_count(&views[2]) != n_groups || k_fft < 2) {
        release_f64_buffers(views, 4);
        PyErr_SetString(PyExc_ValueError,
            "depths and multiplicities must be the same length, and the "
            "output must be the float64 view of an (n_atoms, k_fft) "
            "complex128 array");
        return NULL;
    }
    if (n_atoms > UINT32_MAX || n_groups > UINT32_MAX || k_fft > UINT32_MAX) {
        release_f64_buffers(views, 4);
        PyErr_SetString(PyExc_OverflowError, "batch exceeds uint32 limit");
        return NULL;
    }

    /* Release the GIL: this is n_atoms * n_groups * k_fft/2 complex powers,
     * which reaches tens of seconds on a foundation-scale cohort. */
    Py_BEGIN_ALLOW_THREADS
    err = beam_publicness_pgf((const double *)views[0].buf, (uint32_t)n_atoms,
                             (const double *)views[1].buf,
                             (const double *)views[2].buf,
                             (uint32_t)n_groups, (uint32_t)k_fft,
                             (double *)views[3].buf);
    Py_END_ALLOW_THREADS

    release_f64_buffers(views, 4);
    if (err != BEAM_OK) return set_beam_error(err);
    Py_RETURN_NONE;
}

/* ── publicness_accumulate(pmf, mean, var, weight, edges,
 *                          counts, retained, sigma, floor, tol) → None ── */

static PyObject *py_publicness_accumulate(PyObject *self, PyObject *args) {
    (void)self;
    static const int writable[7] = {0, 0, 0, 0, 0, 1, 1};
    static const char *names[7] = {"pmf", "mean", "variance", "weight",
                                   "bin edges", "count output",
                                   "retained-mass output"};
    PyObject *objs[7];
    Py_buffer views[7];
    Py_ssize_t n_atoms, k_fft, n_bins, pmf_count;
    double tail_sigma, tail_floor, mass_tol;
    BeamError err;

    if (!PyArg_ParseTuple(args, "OOOOOOOddd", &objs[0], &objs[1], &objs[2],
                          &objs[3], &objs[4], &objs[5], &objs[6],
                          &tail_sigma, &tail_floor, &mass_tol)) return NULL;
    if (!get_f64_buffers(objs, views, writable, names, 7)) return NULL;

    n_atoms = f64_buffer_count(&views[1]);
    pmf_count = f64_buffer_count(&views[0]);
    k_fft = (n_atoms > 0 && pmf_count % n_atoms == 0) ? pmf_count / n_atoms : -1;
    n_bins = f64_buffer_count(&views[4]) - 1;

    if (k_fft < 1 || n_bins < 1 ||
        f64_buffer_count(&views[2]) != n_atoms ||
        f64_buffer_count(&views[3]) != n_atoms ||
        f64_buffer_count(&views[6]) != n_atoms ||
        f64_buffer_count(&views[5]) != n_bins) {
        release_f64_buffers(views, 7);
        PyErr_SetString(PyExc_ValueError,
            "pmf must be (n_atoms, k_fft); mean, variance, weight and the "
            "retained-mass output must each have n_atoms entries; and the "
            "count output must have one entry fewer than the bin edges");
        return NULL;
    }
    if (n_atoms > UINT32_MAX || k_fft > UINT32_MAX || n_bins > UINT32_MAX) {
        release_f64_buffers(views, 7);
        PyErr_SetString(PyExc_OverflowError, "batch exceeds uint32 limit");
        return NULL;
    }

    Py_BEGIN_ALLOW_THREADS
    err = beam_publicness_accumulate((const double *)views[0].buf,
                                    (uint32_t)n_atoms, (uint32_t)k_fft,
                                    (const double *)views[1].buf,
                                    (const double *)views[2].buf,
                                    (const double *)views[3].buf,
                                    (const double *)views[4].buf,
                                    (uint32_t)n_bins,
                                    tail_sigma, tail_floor, mass_tol,
                                    (double *)views[5].buf,
                                    (double *)views[6].buf);
    Py_END_ALLOW_THREADS

    release_f64_buffers(views, 7);
    if (err != BEAM_OK) return set_beam_error(err);
    Py_RETURN_NONE;
}

/* ── publicness_accumulate_grouped(..., groups, ..., matrix, ...) → None ── */

static PyObject *py_publicness_accumulate_grouped(
    PyObject *self, PyObject *args) {
    (void)self;
    static const int writable[7] = {0, 0, 0, 0, 0, 1, 1};
    static const char *names[7] = {
        "pmf", "mean", "variance", "weight", "bin edges",
        "grouped count output", "retained-mass output"
    };
    PyObject *objs[7], *group_obj;
    Py_buffer views[7], group_view;
    Py_ssize_t n_atoms, k_fft, n_bins, n_groups_out, pmf_count;
    double tail_sigma, tail_floor, mass_tol;
    BeamError err;

    if (!PyArg_ParseTuple(args, "OOOOOOOOddd",
                          &objs[0], &objs[1], &objs[2], &objs[3],
                          &group_obj, &objs[4], &objs[5], &objs[6],
                          &tail_sigma, &tail_floor, &mass_tol))
        return NULL;
    if (!get_f64_buffers(objs, views, writable, names, 7)) return NULL;
    if (!get_u32_buffer(group_obj, &group_view, "output groups")) {
        release_f64_buffers(views, 7);
        return NULL;
    }

    n_atoms = f64_buffer_count(&views[1]);
    pmf_count = f64_buffer_count(&views[0]);
    k_fft = (n_atoms > 0 && pmf_count % n_atoms == 0)
        ? pmf_count / n_atoms : -1;
    n_bins = f64_buffer_count(&views[4]) - 1;
    n_groups_out = n_bins > 0 && f64_buffer_count(&views[5]) % n_bins == 0
        ? f64_buffer_count(&views[5]) / n_bins : -1;

    if (k_fft < 1 || n_bins < 1 || n_groups_out < 1 ||
        f64_buffer_count(&views[2]) != n_atoms ||
        f64_buffer_count(&views[3]) != n_atoms ||
        group_view.len / (Py_ssize_t)sizeof(uint32_t) != n_atoms ||
        f64_buffer_count(&views[6]) != n_atoms) {
        PyBuffer_Release(&group_view);
        release_f64_buffers(views, 7);
        PyErr_SetString(PyExc_ValueError,
            "pmf must be (n_atoms, k_fft); mean, variance, weight, groups "
            "and retained mass must each have n_atoms entries; and grouped "
            "counts must be a whole number of rows by n_bins");
        return NULL;
    }
    if (n_atoms > UINT32_MAX || k_fft > UINT32_MAX ||
        n_bins > UINT32_MAX || n_groups_out > UINT32_MAX) {
        PyBuffer_Release(&group_view);
        release_f64_buffers(views, 7);
        PyErr_SetString(PyExc_OverflowError, "batch exceeds uint32 limit");
        return NULL;
    }

    Py_BEGIN_ALLOW_THREADS
    err = beam_publicness_accumulate_grouped(
        (const double *)views[0].buf,
        (uint32_t)n_atoms, (uint32_t)k_fft,
        (const double *)views[1].buf,
        (const double *)views[2].buf,
        (const double *)views[3].buf,
        (const uint32_t *)group_view.buf, (uint32_t)n_groups_out,
        (const double *)views[4].buf, (uint32_t)n_bins,
        tail_sigma, tail_floor, mass_tol,
        (double *)views[5].buf, (double *)views[6].buf);
    Py_END_ALLOW_THREADS

    PyBuffer_Release(&group_view);
    release_f64_buffers(views, 7);
    if (err != BEAM_OK) return set_beam_error(err);
    Py_RETURN_NONE;
}

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
    {"n_cells",            py_n_cells,                       METH_O, NULL},
    {"n_nonzero_cells",    py_n_nonzero_cells,               METH_O, NULL},
    {"memory_bytes",       py_memory_bytes,                  METH_O, NULL},
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
    {"surprisal_moments",  py_surprisal_moments,             METH_VARARGS, NULL},
    {"surprisal_cumulants", py_surprisal_cumulants,          METH_VARARGS, NULL},
    {"surprisal_range",    py_surprisal_range,               METH_O, NULL},
    {"length_marginals",   py_length_marginals,              METH_O, NULL},
    {"top_k",              py_top_k,                         METH_VARARGS, NULL},
    {"surprisal_tilt",     py_surprisal_tilt,                METH_VARARGS, NULL},
    {"surprisal_by_depth", py_surprisal_by_depth,            METH_VARARGS, NULL},
    {"topology",           py_topology,                      METH_VARARGS, NULL},
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
