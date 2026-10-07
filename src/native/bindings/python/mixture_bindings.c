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

/**
 * The exponents behind a sequence of floats, or NULL with an exception set.
 *
 * `k` is the member count they must match. The library validates the VALUES;
 * this only has to land them in a C array of the right length, and say which
 * entry was not a number if one was not.
 */
static double *borrow_exponents(PyObject *seq, uint32_t k) {
    PyObject *fast = PySequence_Fast(seq, "expected a sequence of exponents");
    if (!fast) return NULL;
    const Py_ssize_t n = PySequence_Fast_GET_SIZE(fast);
    if (n != (Py_ssize_t)k) {
        Py_DECREF(fast);
        PyErr_Format(PyExc_ValueError,
                     "one exponent per member is needed: %u members and %zd "
                     "exponents", (unsigned)k, n);
        return NULL;
    }
    double *a = (double *)PyMem_Malloc((size_t)k * sizeof(double));
    if (!a) { Py_DECREF(fast);
              return beam_py_nomem("the exponents",
                                   (uint64_t)k * sizeof(double)); }
    for (Py_ssize_t i = 0; i < n; ++i) {
        const double v =
            PyFloat_AsDouble(PySequence_Fast_GET_ITEM(fast, i));
        if (v == -1.0 && PyErr_Occurred()) {
            PyMem_Free(a); Py_DECREF(fast);
            PyErr_Format(PyExc_TypeError,
                         "exponent %zd is not a number", i);
            return NULL;
        }
        a[i] = v;
    }
    Py_DECREF(fast);
    return a;
}

PyObject *py_mixture_power_position(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *seq, *exp_seq;
    unsigned int len = 0;
    if (!PyArg_ParseTuple(args, "OOI", &seq, &exp_seq, &len)) return NULL;
    uint32_t n = 0;
    const BeamModel **v = borrow_models(seq, &n);
    if (!v) return NULL;
    double *a = NULL;
    if (exp_seq != Py_None) {
        a = borrow_exponents(exp_seq, n);
        if (!a) { PyMem_Free(v); return NULL; }
    }

    /* `len` entries, and never zero: a zero-size allocation may legitimately
     * return NULL and would be misread as exhaustion. */
    const size_t want = len ? (size_t)len : 1u;
    double *by = (double *)PyMem_Malloc(want * sizeof(double));
    if (!by) { PyMem_Free(v); PyMem_Free(a);
               return beam_py_nomem("the per-position results",
                                    (uint64_t)want * sizeof(double)); }

    double total = 0.0, sink = 0.0, konst = 0.0;
    BeamError err;
    Py_BEGIN_ALLOW_THREADS
    err = beam_mixture_log_power_product_by_position(v, a, n, (uint32_t)len,
                                                     by, &sink, &konst,
                                                     &total);
    Py_END_ALLOW_THREADS
    PyMem_Free(v); PyMem_Free(a);
    if (err != BEAM_OK) { PyMem_Free(by); return set_beam_error(err); }

    PyObject *b = edge_doubles_to_bytes(by, len);
    PyMem_Free(by);
    if (!b) return NULL;
    return Py_BuildValue("(dNdd)", total, b, sink, konst);
}


PyObject *py_mixture_power_product(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *seq, *exp_seq;
    if (!PyArg_ParseTuple(args, "OO", &seq, &exp_seq)) return NULL;
    uint32_t n = 0;
    const BeamModel **v = borrow_models(seq, &n);
    if (!v) return NULL;
    double *a = borrow_exponents(exp_seq, n);
    if (!a) { PyMem_Free(v); return NULL; }

    const uint32_t slots = len_slots(v[0]);
    double *by = (double *)PyMem_Malloc((size_t)slots * sizeof(double));
    if (!by) { PyMem_Free(v); PyMem_Free(a);
               return beam_py_nomem("the per-length results",
                                    (uint64_t)slots * sizeof(double)); }

    double total = 0.0;
    BeamError err;
    Py_BEGIN_ALLOW_THREADS
    err = beam_mixture_log_power_product_by_length(v, a, n, by, &total);
    Py_END_ALLOW_THREADS
    PyMem_Free(v); PyMem_Free(a);
    if (err != BEAM_OK) { PyMem_Free(by); return set_beam_error(err); }

    PyObject *b = edge_doubles_to_bytes(by, slots);
    PyMem_Free(by);
    if (!b) return NULL;
    return Py_BuildValue("(dN)", total, b);
}

PyObject *py_mixture_power_gram(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *seq;
    double a = 0.0;
    if (!PyArg_ParseTuple(args, "Od", &seq, &a)) return NULL;
    uint32_t n = 0;
    const BeamModel **v = borrow_models(seq, &n);
    if (!v) return NULL;

    double *g = (double *)beam_py_alloc((uint64_t)n * n * sizeof(double),
                                        "the Gram matrix", n, "members");
    if (!g) { PyMem_Free(v); return NULL; }

    BeamError err;
    Py_BEGIN_ALLOW_THREADS
    err = beam_mixture_power_gram(v, n, a, g);
    Py_END_ALLOW_THREADS
    PyMem_Free(v);
    if (err != BEAM_OK) { PyMem_Free(g); return set_beam_error(err); }

    PyObject *b = edge_doubles_to_bytes(g, (size_t)n * n);
    PyMem_Free(g);
    return b;
}

PyObject *py_mixture_power_pairs(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *seq;
    double a = 0.0;
    Py_buffer pb;
    /* The pair list arrives as a buffer of uint32 rather than a Python
     * sequence: a cohort study names thousands of pairs, and converting them
     * one `PyLong` at a time inside the GIL is the only part of this call that
     * would not scale. `y*` accepts any read-only buffer, which a
     * `numpy.uint32` array satisfies. */
    if (!PyArg_ParseTuple(args, "Ody*", &seq, &a, &pb)) return NULL;
    if (pb.len % (Py_ssize_t)(2 * sizeof(uint32_t)) != 0) {
        PyBuffer_Release(&pb);
        PyErr_Format(PyExc_ValueError,
                     "the pair list is %zd bytes, which is not a whole number "
                     "of uint32 pairs", pb.len);
        return NULL;
    }
    const uint64_t n_pairs = (uint64_t)pb.len / (2 * sizeof(uint32_t));

    uint32_t n = 0;
    const BeamModel **v = borrow_models(seq, &n);
    if (!v) { PyBuffer_Release(&pb); return NULL; }

    double *out = (double *)beam_py_alloc(
        (uint64_t)(n_pairs ? n_pairs : 1) * sizeof(double),
        "the pairwise results", (uint32_t)n_pairs, "pairs");
    if (!out) { PyMem_Free(v); PyBuffer_Release(&pb); return NULL; }

    BeamError err;
    Py_BEGIN_ALLOW_THREADS
    err = beam_mixture_power_pairs(v, n, a, (const uint32_t *)pb.buf,
                                   n_pairs, out);
    Py_END_ALLOW_THREADS
    PyMem_Free(v);
    PyBuffer_Release(&pb);
    if (err != BEAM_OK) { PyMem_Free(out); return set_beam_error(err); }

    PyObject *b = edge_doubles_to_bytes(out, (size_t)n_pairs);
    PyMem_Free(out);
    return b;
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
