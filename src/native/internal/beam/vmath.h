/**
 * @file vmath.h
 * @brief exp, eight lanes at a time, for the sweeps' inner loops.
 *
 * Inline so a sweep's loop can take its edge factors exp(q w - qmax) without
 * a call per order; `beam_exp_many` (math/vtrig.c) is the same code behind a
 * function, which is what tests/native/test_vtrig.c holds to long double.
 *
 * ACCURACY. Within two units in the last place for x down to the smallest
 * normal double's logarithm (-708.39); below that the value is flushed to 0,
 * where libm would return a subnormal. A sweep's factors are normalised so
 * the largest is 1, so what is flushed is below 1e-308 of the largest term.
 * exp(0) is exactly 1, which is what keeps M(0) a sum of counts.
 *
 * METHOD. n = round(x / ln 2), r = x - n ln 2 in two parts (the first with
 * trailing zeros, so n times it is exact), a Taylor polynomial of degree 13
 * in r on |r| <= ln 2 / 2 (the first dropped term is below 1e-17), and the
 * power of two put straight into the exponent bits.
 *
 * THE SAME BITS EVERYWHERE. Plain multiplies and adds, no fused
 * multiply-add, each lane on its own: a value does not depend on the CPU the
 * copy was compiled for, nor on its neighbours.
 */
#ifndef BEAM_INTERNAL_VMATH_H
#define BEAM_INTERNAL_VMATH_H

typedef double    vm_v8d __attribute__((vector_size(64)));
typedef long long vm_v8i __attribute__((vector_size(64)));

#define VM_LOG2E   1.44269504088896338700e+00
#define VM_LN2_HI  6.93147180369123816490e-01     /* 32 significant bits */
#define VM_LN2_LO  1.90821492927058770002e-10
#define VM_TOINT   6755399441055744.0             /* 1.5 * 2^52 */
#define VM_EXP_MIN (-708.3964185322641)           /* ln of the smallest normal */

/* *y = exp(*x). Through pointers, not by value: a function passing a vector
 * by value changes the ABI between builds with and without AVX-512. */
__attribute__((always_inline))
static inline void vm_exp8(const vm_v8d *xp, vm_v8d *yp) {
    const vm_v8d x = *xp;
    const vm_v8d lim = {VM_EXP_MIN, VM_EXP_MIN, VM_EXP_MIN, VM_EXP_MIN,
                        VM_EXP_MIN, VM_EXP_MIN, VM_EXP_MIN, VM_EXP_MIN};
    const vm_v8i under = x < lim;                  /* lanes flushed to 0 */
    const vm_v8d xc = (vm_v8d)((under & (vm_v8i)lim) | (~under & (vm_v8i)x));
    /* Rounding by adding 1.5 * 2^52 leaves the integer n in the sum's low
     * bits, so n is read back by subtracting bit patterns: no conversion
     * instruction, which AVX2 and AVX-512F lack for 64-bit integers, and
     * without which the compiler converts the lanes one at a time. */
    const vm_v8d kd = xc * VM_LOG2E + VM_TOINT;
    const vm_v8d fn = kd - VM_TOINT;
    const vm_v8d r = (xc - fn * VM_LN2_HI) - fn * VM_LN2_LO;
    vm_v8d p = r * (1.0 / 6227020800.0) + (1.0 / 479001600.0);     /* 1/13!, 1/12! */
    p = p * r + (1.0 / 39916800.0);
    p = p * r + (1.0 / 3628800.0);
    p = p * r + (1.0 / 362880.0);
    p = p * r + (1.0 / 40320.0);
    p = p * r + (1.0 / 5040.0);
    p = p * r + (1.0 / 720.0);
    p = p * r + (1.0 / 120.0);
    p = p * r + (1.0 / 24.0);
    p = p * r + (1.0 / 6.0);
    p = p * r + 0.5;
    p = p * r + 1.0;
    p = p * r + 1.0;
    const vm_v8d toint = {VM_TOINT, VM_TOINT, VM_TOINT, VM_TOINT,
                          VM_TOINT, VM_TOINT, VM_TOINT, VM_TOINT};
    const vm_v8i n = (vm_v8i)kd - (vm_v8i)toint;
    const vm_v8d two_n = (vm_v8d)((n + 1023) << 52);
    const vm_v8d y = p * two_n;
    *yp = (vm_v8d)(~under & (vm_v8i)y);
}

#endif /* BEAM_INTERNAL_VMATH_H */
