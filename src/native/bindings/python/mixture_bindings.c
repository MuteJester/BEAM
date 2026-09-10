/**
 * @file mixture_bindings.c
 * @brief Entry points for the probability mixture of several fitted models.
 *
 * Each of these takes a SEQUENCE of model capsules rather than one, which is
 * the only structural difference from the rest of the bindings. The borrowed
 * pointer array is built once per call and the models are kept alive by the
 * caller's own references for the duration, which the GIL guarantees across
 * the `Py_BEGIN_ALLOW_THREADS` region: the Python `Mixture` object holds the
 * models it was built from, so nothing can collect one mid-sweep.
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>

#include "beam/common.h"
#include "beam/model.h"
#include "beam/mixture.h"
#include "beam/python_bindings.h"
#include "guards.h"


/**
 * Borrow the models behind a sequence of capsules.
 *
 * Returns NULL with a Python exception set. The caller frees the array with
 * `PyMem_Free`; the models themselves are borrowed and must not be released.
 */
static const BeamModel **borrow_models(PyObject *seq, uint32_t *out_n) {
    PyObject *fast = PySequence_Fast(seq, "expected a sequence of models");
    if (!fast) return NULL;
    const Py_ssize_t n = PySequence_Fast_GET_SIZE(fast);
    if (n <= 0) {
        Py_DECREF(fast);
        PyErr_SetString(PyExc_ValueError,
                        "a mixture needs at least one member, got none");
        return NULL;
    }
    /* The union's 64-member bitmask is enforced by the union sweep itself, not
     * here: this helper serves every entry point, and most of them have no
     * such bound. */
    if ((uint64_t)n > (uint64_t)BEAM_MIXTURE_MAX_PRODUCT_MEMBERS) {
        Py_DECREF(fast);
        PyErr_Format(PyExc_ValueError,
                     "a mixture holds at most %u members and %zd were given",
                     (unsigned)BEAM_MIXTURE_MAX_PRODUCT_MEMBERS, n);
        return NULL;
    }
    const BeamModel **v =
        (const BeamModel **)PyMem_Malloc((size_t)n * sizeof(*v));
    if (!v) { Py_DECREF(fast);
              return beam_py_nomem("the member list",
                                   (uint64_t)n * sizeof(*v)); }
    for (Py_ssize_t i = 0; i < n; ++i) {
        BeamModel *m = edge_from_capsule(PySequence_Fast_GET_ITEM(fast, i));
        if (!m) {
            PyMem_Free(v); Py_DECREF(fast);
            /* `edge_from_capsule` has already set the capsule error, which
             * names neither the argument nor which member failed. */
            PyErr_Format(PyExc_TypeError,
                         "mixture member %zd is not a BEAM model", i);
            return NULL;
        }
        v[i] = m;
    }
    Py_DECREF(fast);
    *out_n = (uint32_t)n;
    return v;
}

/** Slots a by-length vector needs, from any member: they share `max_length`. */
static uint32_t len_slots(const BeamModel *m) {
    return (uint32_t)beam_model_config(m)->max_length + 1u;
}

PyObject *py_mixture_check_shape(PyObject *self, PyObject *arg) {
    (void)self;
    uint32_t n = 0;
    const BeamModel **v = borrow_models(arg, &n);
    if (!v) return NULL;
    const BeamError err = beam_mixture_check_shape(v, n);
    PyMem_Free(v);
    if (err != BEAM_OK) return set_beam_error(err);
    Py_RETURN_NONE;
}

PyObject *py_mixture_inner_product(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *seq;
    if (!PyArg_ParseTuple(args, "O", &seq)) return NULL;
    uint32_t n = 0;
    const BeamModel **v = borrow_models(seq, &n);
    if (!v) return NULL;

    const uint32_t slots = len_slots(v[0]);
    double *by = (double *)PyMem_Malloc((size_t)slots * sizeof(double));
    if (!by) { PyMem_Free(v);
               return beam_py_nomem("the per-length results",
                                    (uint64_t)slots * sizeof(double)); }

    double total = 0.0;
    BeamError err;
    Py_BEGIN_ALLOW_THREADS
    err = beam_mixture_log_inner_product_by_length(v, n, by, &total);
    Py_END_ALLOW_THREADS
    PyMem_Free(v);
    if (err != BEAM_OK) { PyMem_Free(by); return set_beam_error(err); }

    PyObject *b = edge_doubles_to_bytes(by, slots);
    PyMem_Free(by);
    if (!b) return NULL;
    return Py_BuildValue("(dN)", total, b);
}

PyObject *py_mixture_support_union(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *seq;
    unsigned long long cap = 0ull;
    if (!PyArg_ParseTuple(args, "OK", &seq, &cap)) return NULL;
    uint32_t n = 0;
    const BeamModel **v = borrow_models(seq, &n);
    if (!v) return NULL;

    const uint32_t slots = len_slots(v[0]);
    double *by = (double *)PyMem_Malloc((size_t)slots * sizeof(double));
    if (!by) { PyMem_Free(v);
               return beam_py_nomem("the per-length results",
                                    (uint64_t)slots * sizeof(double)); }

    double total = 0.0;
    uint64_t peak = 0;
    BeamError err;
    Py_BEGIN_ALLOW_THREADS
    err = beam_mixture_support_union_by_length(v, n, (uint64_t)cap, by,
                                               &total, &peak);
    Py_END_ALLOW_THREADS
    PyMem_Free(v);
    if (err != BEAM_OK) { PyMem_Free(by); return set_beam_error(err); }

    PyObject *b = edge_doubles_to_bytes(by, slots);
    PyMem_Free(by);
    if (!b) return NULL;
    return Py_BuildValue("(dNK)", total, b, (unsigned long long)peak);
}

PyObject *py_mixture_gram(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *seq;
    if (!PyArg_ParseTuple(args, "O", &seq)) return NULL;
    uint32_t n = 0;
    const BeamModel **v = borrow_models(seq, &n);
    if (!v) return NULL;

    /* n^2 doubles. At the 65,536-member ceiling that is 34 GB, asked for
     * without a word and reported as a bare MemoryError. */
    double *g = (double *)beam_py_alloc((uint64_t)n * n * sizeof(double),
                                        "the Gram matrix", n, "members");
    if (!g) { PyMem_Free(v); return NULL; }

    BeamError err;
    Py_BEGIN_ALLOW_THREADS
    err = beam_mixture_gram(v, n, g);
    Py_END_ALLOW_THREADS
    PyMem_Free(v);
    if (err != BEAM_OK) { PyMem_Free(g); return set_beam_error(err); }

    PyObject *b = edge_doubles_to_bytes(g, (size_t)n * n);
    PyMem_Free(g);
    return b;
}
