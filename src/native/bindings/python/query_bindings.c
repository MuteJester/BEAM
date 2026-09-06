/**
 * @file query_bindings.c
 * @brief Python entry points for the exact probabilistic query engine.
 *
 * The C engine's constraint and automaton are opaque handles with manual
 * lifetimes. None of that crosses into Python. A query instead arrives as a
 * DESCRIPTION -- an array of per-position residue masks, plus an optional
 * nested tuple describing how to compose an automaton -- and this file builds
 * the handles, runs the query and frees them before returning.
 *
 * That is deliberate. Handing capsules to Python would put constraint
 * lifetimes under a garbage collector, for objects whose construction costs
 * microseconds against a dynamic program that costs milliseconds. It would
 * also duplicate the automaton algebra, since a Python layer holding capsules
 * still cannot compose them without calling back in. Rebuilding from a
 * description keeps one implementation of the algebra, in C, and gives Python
 * nothing to leak.
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>

#include "beam/common.h"
#include "beam/model.h"
#include "beam/query.h"

#include "beam/python_bindings.h"

/* ── Building a constraint from a description ────────────────── */

/**
 * One node of an automaton description, recursively.
 *
 * `("motif", (r, ...))`, `("count", of, lo, hi)`, `("masks", (m, ...))`,
 * `("and", a, b)`, `("or", a, b)`, `("not", a)`. Returns NULL with a Python
 * exception set, or with the BEAM error already converted.
 */
static BeamAutomaton *build_automaton(PyObject *spec, uint32_t length,
                                      uint32_t n_residues) {
    if (!PyTuple_Check(spec) || PyTuple_GET_SIZE(spec) < 1) {
        PyErr_SetString(PyExc_TypeError,
                        "an automaton description must be a non-empty tuple");
        return NULL;
    }
    PyObject *kind_o = PyTuple_GET_ITEM(spec, 0);
    const char *kind = PyUnicode_AsUTF8(kind_o);
    if (!kind) return NULL;
    const Py_ssize_t n = PyTuple_GET_SIZE(spec);

    if (strcmp(kind, "motif") == 0 && n == 2) {
        PyObject *seq = PyTuple_GET_ITEM(spec, 1);
        const Py_ssize_t k = PySequence_Size(seq);
        if (k < 0) return NULL;
        uint8_t *res = (uint8_t *)PyMem_Malloc((size_t)(k > 0 ? k : 1));
        if (!res) { PyErr_NoMemory(); return NULL; }
        for (Py_ssize_t i = 0; i < k; ++i) {
            PyObject *it = PySequence_GetItem(seq, i);
            if (!it) { PyMem_Free(res); return NULL; }
            const long v = PyLong_AsLong(it);
            Py_DECREF(it);
            if (v == -1 && PyErr_Occurred()) { PyMem_Free(res); return NULL; }
            res[i] = (uint8_t)v;
        }
        BeamAutomaton *a = beam_query_automaton_motif(length, n_residues, res,
                                                      (uint32_t)k);
        PyMem_Free(res);
        if (!a) { set_beam_error(beam_last_error()); return NULL; }
        return a;
    }
    if (strcmp(kind, "count") == 0 && n == 4) {
        const unsigned long of = PyLong_AsUnsignedLong(PyTuple_GET_ITEM(spec, 1));
        const unsigned long lo = PyLong_AsUnsignedLong(PyTuple_GET_ITEM(spec, 2));
        const unsigned long hi = PyLong_AsUnsignedLong(PyTuple_GET_ITEM(spec, 3));
        if (PyErr_Occurred()) return NULL;
        BeamAutomaton *a = beam_query_automaton_count(
            length, n_residues, (BeamResidueMask)of,
            (uint32_t)lo, (uint32_t)hi);
        if (!a) { set_beam_error(beam_last_error()); return NULL; }
        return a;
    }
    if (strcmp(kind, "masks") == 0 && n == 2) {
        /* A per-position restriction lifted so it can be complemented or
         * disjoined, which a mask cannot be in place. */
        BeamConstraint *c = beam_query_constraint_new(length, n_residues);
        if (!c) { set_beam_error(beam_last_error()); return NULL; }
        PyObject *seq = PyTuple_GET_ITEM(spec, 1);
        for (uint32_t i = 0; i < length; ++i) {
            PyObject *it = PySequence_GetItem(seq, (Py_ssize_t)i);
            if (!it) { beam_query_constraint_free(c); return NULL; }
            const unsigned long v = PyLong_AsUnsignedLong(it);
            Py_DECREF(it);
            if (PyErr_Occurred()) { beam_query_constraint_free(c); return NULL; }
            if (beam_query_constrain_set(c, i, (BeamResidueMask)v) != BEAM_OK) {
                beam_query_constraint_free(c);
                return set_beam_error(beam_last_error()), NULL;
            }
        }
        BeamAutomaton *a = beam_query_automaton_from_masks(c);
        beam_query_constraint_free(c);
        if (!a) { set_beam_error(beam_last_error()); return NULL; }
        return a;
    }
    if ((strcmp(kind, "and") == 0 || strcmp(kind, "or") == 0) && n == 3) {
        BeamAutomaton *x = build_automaton(PyTuple_GET_ITEM(spec, 1), length,
                                           n_residues);
        if (!x) return NULL;
        BeamAutomaton *y = build_automaton(PyTuple_GET_ITEM(spec, 2), length,
                                           n_residues);
        if (!y) { beam_query_automaton_free(x); return NULL; }
        BeamAutomaton *r = (kind[0] == 'a')
                         ? beam_query_automaton_and(x, y)
                         : beam_query_automaton_or(x, y);
        beam_query_automaton_free(x);
        beam_query_automaton_free(y);
        if (!r) { set_beam_error(beam_last_error()); return NULL; }
        return r;
    }
    if (strcmp(kind, "not") == 0 && n == 2) {
        BeamAutomaton *x = build_automaton(PyTuple_GET_ITEM(spec, 1), length,
                                           n_residues);
        if (!x) return NULL;
        BeamAutomaton *r = beam_query_automaton_not(x);
        beam_query_automaton_free(x);
        if (!r) { set_beam_error(beam_last_error()); return NULL; }
        return r;
    }
    PyErr_Format(PyExc_ValueError,
                 "%R is not an automaton description this engine builds", spec);
    return NULL;
}

/**
 * The whole constraint: `length` masks, and an automaton if `spec` is not None.
 *
 * On failure a Python exception is set and NULL returned, so every caller can
 * simply propagate.
 */
static BeamConstraint *build_constraint(BeamModel *m, unsigned int length,
                                        PyObject *masks, PyObject *spec) {
    const uint32_t R = beam_model_n_residues(m);
    BeamConstraint *c = beam_query_constraint_new((uint32_t)length, R);
    if (!c) { set_beam_error(beam_last_error()); return NULL; }

    for (uint32_t i = 0; i < length; ++i) {
        PyObject *it = PySequence_GetItem(masks, (Py_ssize_t)i);
        if (!it) { beam_query_constraint_free(c); return NULL; }
        const unsigned long v = PyLong_AsUnsignedLong(it);
        Py_DECREF(it);
        if (PyErr_Occurred()) { beam_query_constraint_free(c); return NULL; }
        if (beam_query_constrain_set(c, i, (BeamResidueMask)v) != BEAM_OK) {
            beam_query_constraint_free(c);
            return set_beam_error(beam_last_error()), NULL;
        }
    }
    if (spec != Py_None) {
        BeamAutomaton *a = build_automaton(spec, (uint32_t)length, R);
        if (!a) { beam_query_constraint_free(c); return NULL; }
        /* Takes ownership on every path, so `a` is not freed here. */
        if (beam_query_constrain_automaton(c, a) != BEAM_OK) {
            beam_query_constraint_free(c);
            return set_beam_error(beam_last_error()), NULL;
        }
    }
    return c;
}

/** Shared preamble: capsule, then constraint. */
#define QUERY_PROLOGUE(args, fmt, ...)                                        \
    (void)self;                                                               \
    PyObject *cap, *masks, *spec;                                             \
    unsigned int length;                                                      \
    if (!PyArg_ParseTuple(args, fmt, &cap, &length, &masks, &spec,            \
                          ##__VA_ARGS__)) return NULL;                        \
    BeamModel *m = edge_from_capsule(cap);                                    \
    if (!m) return NULL;                                                      \
    BeamConstraint *c = build_constraint(m, length, masks, spec);             \
    if (!c) return NULL

/* ── query_probability(cap, length, masks, spec) → float ─────── */

PyObject *py_query_probability(PyObject *self, PyObject *args) {
    QUERY_PROLOGUE(args, "OIOO");
    double lp = 0.0;
    BeamError err;
    Py_BEGIN_ALLOW_THREADS
    err = beam_query_log_probability(m, c, &lp);
    Py_END_ALLOW_THREADS
    beam_query_constraint_free(c);
    if (err != BEAM_OK) return set_beam_error(err);
    return PyFloat_FromDouble(lp);
}

/* ── query_marginals(cap, length, masks, spec) → (bytes, R, lnZ) ─ */

PyObject *py_query_marginals(PyObject *self, PyObject *args) {
    QUERY_PROLOGUE(args, "OIOO");
    const uint32_t R = beam_model_n_residues(m);
    double *out = (double *)PyMem_Malloc((size_t)length * R * sizeof(double));
    if (!out) { beam_query_constraint_free(c); return PyErr_NoMemory(); }
    double logz = 0.0;
    BeamError err;
    Py_BEGIN_ALLOW_THREADS
    err = beam_query_marginals(m, c, out, &logz);
    Py_END_ALLOW_THREADS
    beam_query_constraint_free(c);
    if (err != BEAM_OK) { PyMem_Free(out); return set_beam_error(err); }
    PyObject *b = edge_doubles_to_bytes(out, (size_t)length * R);
    PyMem_Free(out);
    if (!b) return NULL;
    return Py_BuildValue("(NId)", b, (unsigned int)R, logz);
}

/* ── query_entropy(cap, length, masks, spec) → (H, var) ──────── */

PyObject *py_query_entropy(PyObject *self, PyObject *args) {
    QUERY_PROLOGUE(args, "OIOO");
    double h = 0.0, v = 0.0;
    BeamError err;
    Py_BEGIN_ALLOW_THREADS
    err = beam_query_entropy(m, c, &h, &v);
    Py_END_ALLOW_THREADS
    beam_query_constraint_free(c);
    if (err != BEAM_OK) return set_beam_error(err);
    return Py_BuildValue("(dd)", h, v);
}

/* ── query_expect(cap, length, masks, spec, table) → (mean, var, lnZ) ─ */

PyObject *py_query_expect(PyObject *self, PyObject *args) {
    PyObject *table;
    QUERY_PROLOGUE(args, "OIOOO", &table);
    const uint32_t R = beam_model_n_residues(m);

    BeamFunctional f = { BEAM_FUNCTIONAL_LOG_WEIGHT, NULL };
    double *tab = NULL;
    if (table != Py_None) {
        tab = (double *)PyMem_Malloc((size_t)length * R * sizeof(double));
        if (!tab) { beam_query_constraint_free(c); return PyErr_NoMemory(); }
        for (uint32_t k = 0; k < length * R; ++k) {
            PyObject *it = PySequence_GetItem(table, (Py_ssize_t)k);
            if (!it) { PyMem_Free(tab); beam_query_constraint_free(c);
                       return NULL; }
            tab[k] = PyFloat_AsDouble(it);
            Py_DECREF(it);
            if (PyErr_Occurred()) { PyMem_Free(tab);
                                    beam_query_constraint_free(c); return NULL; }
        }
        f.kind = BEAM_FUNCTIONAL_TABLE;
        f.table = tab;
    }
    BeamQueryMoments mo;
    BeamError err;
    Py_BEGIN_ALLOW_THREADS
    err = beam_query_expect(m, c, &f, &mo);
    Py_END_ALLOW_THREADS
    beam_query_constraint_free(c);
    PyMem_Free(tab);
    if (err != BEAM_OK) return set_beam_error(err);
    return Py_BuildValue("(ddd)", mo.mean, mo.variance, mo.log_z);
}

/* ── query_position_entropy(cap, length, masks, spec) → bytes ── */

PyObject *py_query_position_entropy(PyObject *self, PyObject *args) {
    QUERY_PROLOGUE(args, "OIOO");
    double *out = (double *)PyMem_Malloc((size_t)length * sizeof(double));
    if (!out) { beam_query_constraint_free(c); return PyErr_NoMemory(); }
    BeamError err;
    Py_BEGIN_ALLOW_THREADS
    err = beam_query_position_entropy(m, c, out);
    Py_END_ALLOW_THREADS
    beam_query_constraint_free(c);
    if (err != BEAM_OK) { PyMem_Free(out); return set_beam_error(err); }
    PyObject *b = edge_doubles_to_bytes(out, length);
    PyMem_Free(out);
    return b;
}

/* ── query_pair_joint(cap, length, masks, spec, i, j) → (bytes, R) ─ */

PyObject *py_query_pair_joint(PyObject *self, PyObject *args) {
    unsigned int i, j;
    QUERY_PROLOGUE(args, "OIOOII", &i, &j);
    const uint32_t R = beam_model_n_residues(m);
    double *out = (double *)PyMem_Malloc((size_t)R * R * sizeof(double));
    if (!out) { beam_query_constraint_free(c); return PyErr_NoMemory(); }
    BeamError err;
    Py_BEGIN_ALLOW_THREADS
    err = beam_query_pair_joint(m, c, i, j, out);
    Py_END_ALLOW_THREADS
    beam_query_constraint_free(c);
    if (err != BEAM_OK) { PyMem_Free(out); return set_beam_error(err); }
    PyObject *b = edge_doubles_to_bytes(out, (size_t)R * R);
    PyMem_Free(out);
    if (!b) return NULL;
    return Py_BuildValue("(NI)", b, (unsigned int)R);
}

/* ── query_information_matrix(cap, length, masks, spec) → bytes ─ */

PyObject *py_query_information_matrix(PyObject *self, PyObject *args) {
    QUERY_PROLOGUE(args, "OIOO");
    double *out = (double *)PyMem_Malloc((size_t)length * length
                                         * sizeof(double));
    if (!out) { beam_query_constraint_free(c); return PyErr_NoMemory(); }
    BeamError err;
    Py_BEGIN_ALLOW_THREADS
    err = beam_query_information_matrix(m, c, out);
    Py_END_ALLOW_THREADS
    beam_query_constraint_free(c);
    if (err != BEAM_OK) { PyMem_Free(out); return set_beam_error(err); }
    PyObject *b = edge_doubles_to_bytes(out, (size_t)length * length);
    PyMem_Free(out);
    return b;
}

/* ── query_cost(cap, length, masks, spec, whole) → (states, edges, bytes) ─ */

PyObject *py_query_cost(PyObject *self, PyObject *args) {
    int whole;
    QUERY_PROLOGUE(args, "OIOOp", &whole);
    BeamQueryCost cost;
    const BeamError err = whole ? beam_query_information_cost(m, c, &cost)
                                : beam_query_estimate(m, c, &cost);
    beam_query_constraint_free(c);
    if (err != BEAM_OK) return set_beam_error(err);
    return Py_BuildValue("(KKK)", (unsigned long long)cost.states,
                         (unsigned long long)cost.edges,
                         (unsigned long long)cost.bytes);
}

/* ── query_neighbours(cap, sequence, deletions) → list ───────── */

PyObject *py_query_neighbours(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap;
    const char *seq;
    int deletions;
    if (!PyArg_ParseTuple(args, "Osp", &cap, &seq, &deletions)) return NULL;
    BeamModel *m = edge_from_capsule(cap);
    if (!m) return NULL;

    BeamNeighbourhood nb;
    BeamError err;
    Py_BEGIN_ALLOW_THREADS
    err = deletions ? beam_query_deletions(m, seq, &nb)
                    : beam_query_insertions(m, seq, &nb);
    Py_END_ALLOW_THREADS
    if (err != BEAM_OK) return set_beam_error(err);

    PyObject *list = PyList_New(nb.n);
    if (!list) { beam_query_neighbourhood_free(&nb); return NULL; }
    for (uint32_t k = 0; k < nb.n; ++k) {
        PyObject *row = Py_BuildValue(
            "(sIIIdd)", nb.sequences + (size_t)k * (nb.length + 1u),
            (unsigned int)nb.site[k], (unsigned int)nb.residue[k],
            (unsigned int)nb.multiplicity[k],
            nb.log_odds[k], nb.posterior[k]);
        if (!row) { Py_DECREF(list); beam_query_neighbourhood_free(&nb);
                    return NULL; }
        PyList_SET_ITEM(list, k, row);
    }
    const double src = nb.log_source;
    const int producible = nb.length_producible ? 1 : 0;
    beam_query_neighbourhood_free(&nb);
    return Py_BuildValue("(NdO)", list, src,
                         producible ? Py_True : Py_False);
}
