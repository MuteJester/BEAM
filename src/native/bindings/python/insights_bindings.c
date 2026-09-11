/**
 * @file insights_bindings.c
 * @brief The `insights` layer's entry points.
 *
 * One shape decision worth stating: a logo comes back as a tuple of byte
 * blocks rather than as anything numpy-shaped, exactly as the other analytics
 * bindings do. The reshaping is naming -- which axis is position and which is
 * residue -- and naming belongs on the Python side where the names can sit
 * next to what they mean.
 */
#include "beam/python_bindings.h"
#include "beam/insights.h"

/** A `[n]` block of uint32 as bytes, the counterpart of the double helper. */
static PyObject *u32_to_bytes(const uint32_t *v, size_t n) {
    return PyBytes_FromStringAndSize((const char *)v,
                                     (Py_ssize_t)(n * sizeof(uint32_t)));
}

/** One logo as a tuple; NULL with the error set on failure. */
static PyObject *logo_tuple(const BeamLogo *g) {
    const size_t cell = (size_t)g->length * g->n_residues;
    const size_t L = g->length;
    PyObject *pm = g->position_mass
        ? edge_doubles_to_bytes(g->position_mass, L)
        : (Py_INCREF(Py_None), Py_None);
    if (!pm) return NULL;
    /* `resolution_error` rides in as a plain double and may be NaN, which is
     * the value that says no grid was involved. Python turns it into None. */
    return Py_BuildValue(
        "(NNNNNNNNNNNdddddd)",
        edge_doubles_to_bytes(g->probability, cell),
        edge_doubles_to_bytes(g->height_bits, cell),
        edge_doubles_to_bytes(g->height_nats, cell),
        edge_doubles_to_bytes(g->entropy_bits, L),
        edge_doubles_to_bytes(g->entropy_nats, L),
        edge_doubles_to_bytes(g->information_bits, L),
        edge_doubles_to_bytes(g->information_nats, L),
        u32_to_bytes(g->n_support, L),
        u32_to_bytes(g->consensus, L),
        edge_doubles_to_bytes(g->consensus_probability, L),
        pm,
        g->lo_nats, g->hi_nats, g->log_mass, g->mass, g->fraction,
        g->resolution_error);
}


/**
 * `insights_logos(cap, length, edges, counting, bins, spacing,
 *                 reference_residues, max_resolution_error)`
 *
 * `edges` is a bytes buffer of float64 surprisal edges, or None. Infinities
 * are permitted and are resolved by the C layer against the grid, which is
 * the reason this call is worth making at all rather than assembling the
 * bands in Python.
 */
PyObject *py_insights_logos(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap, *edges_obj;
    unsigned int length, bins, ref;
    int counting = 0;
    double spacing, max_res;
    if (!PyArg_ParseTuple(args, "OIOpIdId", &cap, &length, &edges_obj,
                          &counting, &bins, &spacing, &ref, &max_res))
        return NULL;
    BeamModel *m = edge_from_capsule(cap);
    if (!m) return NULL;

    const double *edges = NULL;
    size_t n_edges = 0;
    if (edges_obj != Py_None) {
        char *buf; Py_ssize_t nb;
        if (PyBytes_AsStringAndSize(edges_obj, &buf, &nb) != 0) return NULL;
        if (nb % (Py_ssize_t)sizeof(double) != 0) {
            PyErr_SetString(PyExc_ValueError,
                            "band edges must be a whole number of float64s");
            return NULL;
        }
        edges = (const double *)buf;
        n_edges = (size_t)nb / sizeof(double);
    }

    BeamLogoRequest req;
    beam_logo_request_default(&req);
    req.length = length;
    req.edges = edges;
    req.n_edges = (uint32_t)n_edges;
    req.counting = (bool)counting;
    req.bins = bins;
    req.spacing = spacing;
    req.reference_residues = ref;
    req.max_resolution_error = max_res;

    BeamLogoSet set;
    BeamError err;
    Py_BEGIN_ALLOW_THREADS
    err = beam_insights_logos(m, &req, &set);
    Py_END_ALLOW_THREADS
    if (err != BEAM_OK) return set_beam_error(err);

    PyObject *logos = PyList_New((Py_ssize_t)set.n_logos);
    if (!logos) { beam_insights_logo_set_free(&set); return NULL; }
    for (uint32_t j = 0; j < set.n_logos; ++j) {
        PyObject *t = logo_tuple(&set.logo[j]);
        if (!t) { Py_DECREF(logos); beam_insights_logo_set_free(&set); return NULL; }
        PyList_SET_ITEM(logos, (Py_ssize_t)j, t);
    }
    /* `i` for the flags, not `p`: that format character is for parsing
     * arguments, not for building a value. */
    PyObject *out = Py_BuildValue("(NIIIdddddii)", logos,
                                  (unsigned int)set.logo[0].n_residues,
                                  (unsigned int)set.alphabet,
                                  (unsigned int)set.reference_residues,
                                  set.max_entropy_bits, set.max_entropy_nats,
                                  set.on_grid_total, set.length_total,
                                  set.spacing,
                                  (int)set.exact, (int)set.counting);
    beam_insights_logo_set_free(&set);
    return out;
}

/** `insights_logo_cost(...) -> (states, edges, bytes)`, same arguments. */
PyObject *py_insights_logo_cost(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *cap, *edges_obj;
    unsigned int length, bins, ref;
    int counting = 0;
    double spacing, max_res;
    if (!PyArg_ParseTuple(args, "OIOpIdId", &cap, &length, &edges_obj,
                          &counting, &bins, &spacing, &ref, &max_res))
        return NULL;
    BeamModel *m = edge_from_capsule(cap);
    if (!m) return NULL;

    const double *edges = NULL;
    size_t n_edges = 0;
    if (edges_obj != Py_None) {
        char *buf; Py_ssize_t nb;
        if (PyBytes_AsStringAndSize(edges_obj, &buf, &nb) != 0) return NULL;
        edges = (const double *)buf;
        n_edges = (size_t)nb / sizeof(double);
    }

    BeamLogoRequest req;
    beam_logo_request_default(&req);
    req.length = length; req.edges = edges; req.n_edges = (uint32_t)n_edges;
    req.counting = (bool)counting; req.bins = bins; req.spacing = spacing;
    req.reference_residues = ref; req.max_resolution_error = max_res;

    BeamQueryCost cost;
    const BeamError err = beam_insights_logo_cost(m, &req, &cost);
    if (err != BEAM_OK) return set_beam_error(err);
    return Py_BuildValue("(KKK)", (unsigned long long)cost.states,
                         (unsigned long long)cost.edges,
                         (unsigned long long)cost.bytes);
}
