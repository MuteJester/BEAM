/**
 * @file common.c
 * @brief Error mapping, capsule ownership, and argument conversion.
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

/*
 * Map a C error code onto a Python exception CLASS.
 *
 * A table, not a switch, and the classes come from `BEAM.errors` so a caller
 * can write `except BudgetExceeded` instead of matching on the text of a
 * message. Every class keeps the built-in it replaces as a second base, so
 * this is additive: `except ValueError` catches everything it caught before.
 *
 * THE ENTRY POINT STILL WINS. This table is the default for a code, not a
 * decree. Several bindings deliberately raise something more specific before
 * ever reaching here -- `positional_marginals` turns BEAM_ERR_NO_LIVE_PATHS
 * into a ValueError about the clamp, and the query facade narrows built-ins to
 * its own classes -- because one code does not imply one class. Assuming it
 * did is how BEAM_ERR_IO_CORRUPT silently converted three `beam_model_load`
 * failures from OSError to ValueError: its arm had never run, so nobody had
 * ever seen what it said. `tests/python/test_error_contract.py` pins the
 * observed class per entry point, which is the thing that must not move.
 *
 * The import is LAZY and cached, not done at module init: `BEAM/__init__.py`
 * imports `_cbeam`, so reaching back into `BEAM.errors` from this module's
 * initialisation would be circular. By the time any error is raised the
 * package is fully imported. If the import fails for any reason the built-in
 * is used instead, so a broken install degrades to the old behaviour rather
 * than to a confusing secondary failure.
 */
static PyObject *beam_error_class(BeamError err, PyObject *fallback) {
    static PyObject *mod = NULL;      /* borrowed-forever, module-lifetime */
    const char *name = NULL;

    switch (err) {
        case BEAM_ERR_BUDGET:            name = "BudgetExceeded";  break;
        case BEAM_ERR_ALLOC:             name = "OutOfMemory";     break;

        case BEAM_ERR_INVALID_SEQUENCE:  name = "InvalidSequence"; break;
        case BEAM_ERR_NULL_ARG:          /* == BEAM_ERR_INVALID_ARG */
        case BEAM_ERR_EMPTY_INPUT:
        case BEAM_ERR_INVALID_VARIANT:
        case BEAM_ERR_LENGTH_MISMATCH:
        case BEAM_ERR_PARAM_OUT_OF_RANGE: name = "BeamValueError"; break;

        case BEAM_ERR_NOT_BUILT:         name = "NotFinalized";    break;
        case BEAM_ERR_NO_LIVE_PATHS:     name = "NoLivePaths";     break;
        case BEAM_ERR_VARIANT_MISMATCH:  name = "ModelMismatch";   break;

        case BEAM_ERR_IO_OPEN:
        case BEAM_ERR_IO_READ:
        case BEAM_ERR_IO_WRITE:
        case BEAM_ERR_IO_CORRUPT:
        case BEAM_ERR_IO_VERSION:        name = "ModelFileError";  break;

        /* Library bugs and codes nothing produces. BEAM_ERR_HAS_CYCLES is
         * among them: a cyclic trellis would be an invariant violation, not a
         * user error. */
        case BEAM_ERR_INTERNAL:
        case BEAM_ERR_HAS_CYCLES:
        case BEAM_ERR_OVERFLOW:
        case BEAM_ERR_CONVERGENCE:
        case BEAM_ERR_NO_GENE_DATA:
        case BEAM_ERR_GENE_NOT_FOUND:
        case BEAM_ERR_MISSING_EDGE:      name = "InternalError";   break;

        default:                         return fallback;
    }

    if (!mod) {
        mod = PyImport_ImportModule("BEAM.errors");
        if (!mod) { PyErr_Clear(); return fallback; }
    }
    PyObject *cls = PyObject_GetAttrString(mod, name);
    if (!cls) { PyErr_Clear(); return fallback; }
    return cls;                      /* new reference; caller decrefs */
}

PyObject *set_beam_error(BeamError err) {
    const char *msg = beam_error_message();
    const int has_msg = msg && msg[0];

    /*
     * The built-in each code produced before this table existed. Keeping it as
     * the fallback means a failed import cannot change what a caller catches,
     * and it documents the mapping the exception classes are built to
     * preserve.
     */
    PyObject *builtin;
    const char *generic;
    switch (err) {
        case BEAM_ERR_BUDGET:
        case BEAM_ERR_ALLOC:
            /* PyErr_NoMemory() sets MemoryError with no message, which is
             * right for a bare allocation failure and wrong for a refusal
             * that has something to say -- "this configuration needs 41 GB
             * and the budget is 4" tells a caller what to change. */
            if (!has_msg) return PyErr_NoMemory();
            builtin = PyExc_MemoryError; generic = "out of memory";      break;

        case BEAM_ERR_NULL_ARG:
        case BEAM_ERR_EMPTY_INPUT:
        case BEAM_ERR_INVALID_SEQUENCE:
        case BEAM_ERR_INVALID_VARIANT:
        case BEAM_ERR_LENGTH_MISMATCH:
        case BEAM_ERR_PARAM_OUT_OF_RANGE:
            builtin = PyExc_ValueError;  generic = "invalid argument";   break;

        case BEAM_ERR_NOT_BUILT:
        case BEAM_ERR_NO_LIVE_PATHS:
        case BEAM_ERR_HAS_CYCLES:
            builtin = PyExc_RuntimeError; generic = "model state error"; break;

        case BEAM_ERR_IO_OPEN:
        case BEAM_ERR_IO_READ:
        case BEAM_ERR_IO_WRITE:
        case BEAM_ERR_IO_CORRUPT:
        case BEAM_ERR_IO_VERSION:
            /* OSError for the whole family: that is the contract
             * `beam_model_load` has always had, and three tests depend on it. */
            builtin = PyExc_OSError;      generic = "I/O error";         break;

        default:
            builtin = PyExc_RuntimeError; generic = "unknown error";     break;
    }

    PyObject *cls = beam_error_class(err, builtin);
    PyErr_SetString(cls, has_msg ? msg : generic);
    if (cls != builtin) Py_DECREF(cls);
    return NULL;
}
void capsule_destructor(PyObject *capsule) {
    BeamModel *m = (BeamModel *)PyCapsule_GetPointer(capsule, BEAM_CAPSULE_NAME);
    if (m) beam_model_destroy(m);
}
BeamModel *edge_from_capsule(PyObject *capsule) {
    return (BeamModel *)PyCapsule_GetPointer(capsule, BEAM_CAPSULE_NAME);
}
const char **edge_borrow_strings(PyObject *seq, Py_ssize_t *out_n) {
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
double *edge_borrow_abundances(PyObject *abund, Py_ssize_t n, bool *out_ok) {
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
int edge_parse_config(PyObject *args, BeamModelConfig *cfg,
                             PyObject **seqs, PyObject **abund) {
    int order, depth_cap, max_length, strict, sparse;
    double cw, cm, cl, k0, k1, k2, kw, kl, wf, wb;
    /* The alphabet is appended rather than placed first, where it belongs,
     * so every existing position in the tuple keeps its meaning. */
    const char *alphabet = NULL;
    const char *fmt = seqs ? "OOiii(ddd)(ddd)dd(dd)pps" :
                             "iii(ddd)(ddd)dd(dd)pps";
    int ok;
    if (seqs)
        ok = PyArg_ParseTuple(args, fmt, seqs, abund, &order, &depth_cap,
                              &max_length, &cw, &cm, &cl, &k0, &k1, &k2,
                              &kw, &kl, &wf, &wb, &strict, &sparse, &alphabet);
    else
        ok = PyArg_ParseTuple(args, fmt, &order, &depth_cap, &max_length,
                              &cw, &cm, &cl, &k0, &k1, &k2, &kw, &kl, &wf, &wb,
                              &strict, &sparse, &alphabet);
    if (!ok) return 0;
    const BeamAlphabetInfo *a = beam_alphabet_by_name(alphabet);
    if (!a) {
        PyErr_Format(PyExc_ValueError, "unknown alphabet '%s'", alphabet);
        return 0;
    }
    beam_model_config_default_for(a->id, cfg);
    /*
     * Range-checked BEFORE the cast. These three were cast straight to
     * uint8_t, so `order = 258` wrapped to 2 -- a LEGAL order, which then
     * passed `beam_config_valid` and produced a fully working model at an
     * order the caller never asked for, scoring wrongly and saying nothing.
     * `max_length = 300` wrapped to 44 the same way. The Python facade
     * validates these properly, but `_cbeam` is an importable boundary and
     * must not be the one place where a nonsense argument silently becomes a
     * plausible one.
     */
    if (order < 0 || order > 255 || depth_cap < 0 || depth_cap > 255
        || max_length < 0 || max_length > 255) {
        PyErr_Format(PyExc_ValueError,
                     "order, depth_cap and max_length must each be in 0..255 "
                     "(got %d, %d, %d); a larger value would wrap to an "
                     "unrelated but legal setting",
                     order, depth_cap, max_length);
        return 0;
    }
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
PyObject *edge_doubles_to_bytes(const double *v, size_t n) {
    return PyBytes_FromStringAndSize((const char *)v,
                                     (Py_ssize_t)(n * sizeof(double)));
}
PyObject *edge_moment_list(const BeamModel *m, double q,
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
