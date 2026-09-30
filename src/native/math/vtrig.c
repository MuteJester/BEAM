/**
 * @file vtrig.c
 * @brief Sine and cosine, four at a time, and exp, eight at a time, for the
 *        sweeps.
 *
 * The line sweep needs a few rotations per edge per stretch, and libm's
 * sincos, one angle at a time, had become a third of a line's time on the
 * foundation. This does the same job on whole vectors: a three-part
 * Cody-Waite reduction by pi/2 and Taylor polynomials on [-pi/4, pi/4].
 *
 * ACCURACY. Within two units in the last place of 1 for |x| up to 2^19 pi/2
 * (about 8e5 radians; the sweep's angles stay below about 1e5), which
 * tests/native/test_vtrig.c holds against long double. The reduction is
 * exact in its first step: pi/2's first two parts carry 33 bits each, so
 * n times either is exact for |n| < 2^20, and x - n P1 is exact because the
 * two are within a factor of two. Past that range an angle goes to libm.
 * sin 0 and cos 0 come out exactly 0 and 1.
 *
 * THE SAME BITS EVERYWHERE. Plain multiplies and adds, no fused
 * multiply-add, each lane on its own: an angle's value does not depend on
 * the CPU the copy was picked for, nor on which other angles shared its
 * call.
 */
#include "beam/internal.h"
#include "beam/vmath.h"

typedef double vt_v4d __attribute__((vector_size(32)));
typedef long long vt_v4i __attribute__((vector_size(32)));

#define VT_LOAD(dst, p)  memcpy(&(dst), (p), sizeof(vt_v4d))
#define VT_STORE(p, src) memcpy((p), &(src), sizeof(vt_v4d))

/* pi/2 in three parts: 33 bits, 33 bits, and the remainder. */
#define VT_PIO2_1  1.57079632673412561417e+00
#define VT_PIO2_2  6.07710050630396597660e-11
#define VT_PIO2_2T 2.02226624879595063154e-21
#define VT_INVPIO2 6.36619772367581382433e-01
#define VT_TOINT   6755399441055744.0            /* 1.5 * 2^52 */
#define VT_LIMIT   823549.6654                  /* 2^19 pi/2    */

/* (-1)^k / (2k+1)!  and  (-1)^k / (2k)!, k = 1 .. */
#define VT_S1 -1.66666666666666666667e-01
#define VT_S2  8.33333333333333333333e-03
#define VT_S3 -1.98412698412698412698e-04
#define VT_S4  2.75573192239858906526e-06
#define VT_S5 -2.50521083854417187751e-08
#define VT_S6  1.60590438368216145994e-10
#define VT_S7 -7.64716373181981647590e-13
#define VT_S8  2.81145725434552076320e-15
#define VT_C1 -5.00000000000000000000e-01
#define VT_C2  4.16666666666666666667e-02
#define VT_C3 -1.38888888888888888889e-03
#define VT_C4  2.48015873015873015873e-05
#define VT_C5 -2.75573192239858906526e-07
#define VT_C6  2.08767569878680989792e-09
#define VT_C7 -1.14707455977297247139e-11
#define VT_C8  4.77947733238738529744e-14
#define VT_C9 -1.56192069685862264622e-16

__attribute__((always_inline))
static inline void vt_sincos4(vt_v4d x, vt_v4d *s_out, vt_v4d *c_out) {
    /* n read back from the rounded sum's low bits (see beam/vmath.h): no
     * 64-bit conversion instruction, which AVX2 lacks. */
    const vt_v4d kd = x * VT_INVPIO2 + VT_TOINT;
    const vt_v4d fn = kd - VT_TOINT;
    const vt_v4d toint = {VT_TOINT, VT_TOINT, VT_TOINT, VT_TOINT};
    const vt_v4i n = (vt_v4i)kd - (vt_v4i)toint;
    const vt_v4d r = ((x - fn * VT_PIO2_1) - fn * VT_PIO2_2) - fn * VT_PIO2_2T;
    const vt_v4d z = r * r;
    const vt_v4d ps = VT_S1 + z * (VT_S2 + z * (VT_S3 + z * (VT_S4 + z * (VT_S5
                    + z * (VT_S6 + z * (VT_S7 + z * VT_S8))))));
    const vt_v4d pc = VT_C2 + z * (VT_C3 + z * (VT_C4 + z * (VT_C5 + z * (VT_C6
                    + z * (VT_C7 + z * (VT_C8 + z * VT_C9))))));
    const vt_v4d s = r + r * z * ps;
    const vt_v4d c = 1.0 + z * (VT_C1 + z * pc);
    /* Quadrant n mod 4: (sin, cos) = (s, c), (c, -s), (-s, -c), (-c, s). */
    const vt_v4i one = {1, 1, 1, 1}, two = {2, 2, 2, 2};
    const vt_v4i zero = {0, 0, 0, 0};
    const vt_v4i sign = {(long long)0x8000000000000000ULL, (long long)0x8000000000000000ULL,
                         (long long)0x8000000000000000ULL, (long long)0x8000000000000000ULL};
    const vt_v4i swap = (n & one) != zero;
    const vt_v4i sneg = (n & two) != zero;
    const vt_v4i cneg = ((n + one) & two) != zero;
    const vt_v4i si = (swap & (vt_v4i)c) | (~swap & (vt_v4i)s);
    const vt_v4i ci = (swap & (vt_v4i)s) | (~swap & (vt_v4i)c);
    *s_out = (vt_v4d)(si ^ (sneg & sign));
    *c_out = (vt_v4d)(ci ^ (cneg & sign));
}

#if defined(__x86_64__) && defined(__ELF__) && defined(__GNUC__)
#define VT_CLONES __attribute__((target_clones("avx2", "default")))
#else
#define VT_CLONES
#endif

VT_CLONES
void beam_sincos_many(const double *x, double *s, double *c, size_t n) {
    size_t k = 0;
    for (; k + 4u <= n; k += 4u) {
        vt_v4d xv, sv, cv;
        VT_LOAD(xv, x + k);
        vt_sincos4(xv, &sv, &cv);
        VT_STORE(s + k, sv);
        VT_STORE(c + k, cv);
    }
    if (k < n) {                           /* the rest, as one padded vector */
        double xb[4] = {0.0, 0.0, 0.0, 0.0}, sb[4], cb[4];
        memcpy(xb, x + k, (n - k) * sizeof(double));
        vt_v4d xv, sv, cv;
        VT_LOAD(xv, xb);
        vt_sincos4(xv, &sv, &cv);
        VT_STORE(sb, sv);
        VT_STORE(cb, cv);
        memcpy(s + k, sb, (n - k) * sizeof(double));
        memcpy(c + k, cb, (n - k) * sizeof(double));
    }
    for (k = 0; k < n; ++k)                /* beyond the reduction's range */
        if (!(fabs(x[k]) <= VT_LIMIT)) { s[k] = sin(x[k]); c[k] = cos(x[k]); }
}

/* exp(x[k]) for every k, eight at a time, as vm_exp8 (beam/vmath.h) computes
 * it inside the sweeps; this is its tested entry point. */
VT_CLONES
void beam_exp_many(const double *x, double *y, size_t n) {
    size_t k = 0;
    for (; k + 8u <= n; k += 8u) {
        vm_v8d xv, yv;
        memcpy(&xv, x + k, sizeof xv);
        vm_exp8(&xv, &yv);
        memcpy(y + k, &yv, sizeof yv);
    }
    if (k < n) {                           /* the rest, as one padded vector */
        double xb[8] = {0, 0, 0, 0, 0, 0, 0, 0}, yb[8];
        memcpy(xb, x + k, (n - k) * sizeof(double));
        vm_v8d xv, yv;
        memcpy(&xv, xb, sizeof xv);
        vm_exp8(&xv, &yv);
        memcpy(yb, &yv, sizeof yv);
        memcpy(y + k, yb, (n - k) * sizeof(double));
    }
}
