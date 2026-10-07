#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "beam/python_bindings.h"
#include "beam/fingerprint.h"
#include "guards.h"

PyObject *py_fingerprint_search(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *pa, *qa, *sizes[6];
    BeamFingerprintConfig cfg;
    uint32_t L;
    if (!PyArg_ParseTuple(args,"OOOOOOOOdd", &pa,&qa,&sizes[0],&sizes[1],
                          &sizes[2],&sizes[3],&sizes[4],&sizes[5],
                          &cfg.min_mass,&cfg.min_gain)) return NULL;
    const char *names[] = {"length","max_leaves","max_depth","lookahead","beam_width","max_queries"};
    uint32_t *values[] = {&L,&cfg.max_leaves,&cfg.max_depth,&cfg.lookahead,&cfg.beam_width,&cfg.max_queries};
    for (unsigned i = 0; i < 6; ++i)
        if (beam_py_count(sizes[i],names[i],values[i]) < 0) return NULL;
    BeamModel *p = edge_from_capsule(pa), *q = edge_from_capsule(qa);
    if (!p || !q) return NULL;
    BeamFingerprintResult *r = NULL;
    BeamError err;
    Py_BEGIN_ALLOW_THREADS
    err = beam_fingerprint_search(p,q,L,&cfg,&r);
    Py_END_ALLOW_THREADS
    if (err != BEAM_OK) return set_beam_error(err);
    PyObject *indices = PyBytes_FromStringAndSize(NULL,(Py_ssize_t)r->n_nodes*6*sizeof(int32_t));
    PyObject *masses = PyBytes_FromStringAndSize(NULL,(Py_ssize_t)r->n_nodes*2*sizeof(double));
    if (!indices || !masses) {
        Py_XDECREF(indices); Py_XDECREF(masses); beam_fingerprint_free(r); return NULL;
    }
    /* memcpy avoids alignment assumptions about the bytes object's payload. */
    for (uint32_t i = 0; i < r->n_nodes; ++i) {
        const BeamFingerprintNode *n = r->nodes+i;
        int32_t v[] = {n->parent,n->yes,n->no,n->position,n->residue,(int32_t)n->depth};
        double w[] = {n->p,n->q};
        memcpy(PyBytes_AS_STRING(indices)+(size_t)i*sizeof(v),v,sizeof(v));
        memcpy(PyBytes_AS_STRING(masses)+(size_t)i*sizeof(w),w,sizeof(w));
    }
    PyObject *out = Py_BuildValue("(NNIiKd)", indices,masses,r->queries,
                                  (int)r->budget_exhausted,
                                  (unsigned long long)r->scratch_bytes,r->squared_distance);
    beam_fingerprint_free(r);
    return out;
}
