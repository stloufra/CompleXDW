#ifndef XDWARITH_H
#define XDWARITH_H

#include "XDWbasicOP.h"
#include "XDWerrorFree.h"

namespace XDW_ARTH{

// SloppyDWPlusDW — 11 flops
// Relative error <= 1
FLOAT_TEMPLATE_GUARD
__cuda_callable__
static constexpr __xdw_inline__ void
SloppyDWPlusDW(const T xh, const T xl, const T yh, const T yl, T* __restrict__ zh, T* __restrict__ zl)
{
   // TwoSum(xh, yh, &sh, &sl)
   rne<T> r1 = two_sum(xh, yh);
   // v  = xl + yl
   T v = add_rn(xl, yl);
   // w  = sl + v
   T w = add_rn(r1.error, v);
   // Fast2Sum(sh, w, &zh, &zl)
   rne<T> r2 = quick_two_sum(r1.sum, w);

   *zh = r2.sum; *zl = r2.error;
}

// AccurateDWPlusDW — 20 flops
// Relative error < 3u^2.
FLOAT_TEMPLATE_GUARD
__cuda_callable__
static constexpr __xdw_inline__ void
AccurateDWPlusDW(const T xh, const T xl, const T yh, const T yl, T* __restrict__ zh, T* __restrict__ zl)
{
   // TwoSum(xh, yh, &sh, &sl)
   rne<T> r1 = two_sum(xh, yh);
   // TwoSum(xl, yl, &th, &tl)
   rne<T> r2 = two_sum(xl, yl);
   // c  = sl + th
   T c = add_rn(r1.error, r2.sum);
   // Fast2Sum(sh, c, &vh, &vl)
   rne<T> r3 = quick_two_sum(r1.sum, c);
   // w  = tl + vl
   T w = add_rn(r2.error, r3.error);
   // Fast2Sum(vh, w, &zh, &zl)
   rne<T> r4 = quick_two_sum(r3.sum, w);

   *zh = r4.sum; *zl = r4.error;
}

// maddDWPlusDW — 20 flops
// Relative error 2u^2. 
FLOAT_TEMPLATE_GUARD
__cuda_callable__
static constexpr __xdw_inline__ void
maddDWPlusDW(const T xh, const T xl, const T yh, const T yl, T* __restrict__ zh, T* __restrict__ zl)
{
   // TwoSum(xh, yh, &sh, &sl)
   rne<T> r1 = two_sum(xh, yh);
   // TwoSum(xl, yl, &th, &tl)
   rne<T> r2 = two_sum(xl, yl);
   // Fast2Sum(sh, th, &ah, &al)
   rne<T> r3 = quick_two_sum(r1.sum, r2.sum);
   // b  = sl + tl
   T b = add_rn(r1.error, r2.error);
   // e  = b + al
   T e = add_rn(b, r3.error);
   // Fast2Sum(ah, e, &zh, &zl)
   rne<T> r4 = quick_two_sum(r3.sum, e);

   *zh = r4.sum; *zl = r4.error;
}

// DWTimesDW2 — 8 flops
// Relative error <= 5u^2. 
FLOAT_TEMPLATE_GUARD
__cuda_callable__
static constexpr __xdw_inline__ void
DWTimesDW2(const T xh, const T xl, const T yh, const T yl, T* __restrict__ zh, T* __restrict__ zl)
{
   // TwoProdFMA(xh, yh, &ch, &cl1)
   rne<T> r1 = two_prod(xh, yh);
   // tl   = xh * yl
   T tl = mul_rn(xh, yl);
   // cl2  = fma(xl, yh, tl)
   T cl2 = fma_rn(xl, yh, tl);
   // cl3  = cl1 + cl2
   T cl3 = add_rn(r1.error, cl2);
   // Fast2Sum(ch, cl3, &zh, &zl)
   rne<T> r2 = quick_two_sum(r1.sum, cl3);

   *zh = r2.sum; *zl = r2.error;
}

// DWTimesDWunnorm — 5 flops
// Unnormalized DW product, skips final Fast2Sum. 
// Returns (ph, pl) where pl may not satisfy |pl| <= u*|ph|.
FLOAT_TEMPLATE_GUARD
__cuda_callable__
static constexpr __xdw_inline__ void
DWTimesDWunnorm(const T xh, const T xl, const T yh, const T yl, T* __restrict__ zh, T* __restrict__ zl)
{
   // TwoProdFMA(xh, yh, &ch, &cl1)
   rne<T> r1 = two_prod(xh, yh);
   // t   = xh * yl
   T t = mul_rn(xh, yl);
   // u   = fma(xl, yh, t)
   T u = fma_rn(xl, yh, t);
   // cl3 = cl1 + u
   T cl3 = add_rn(r1.error, u);

   *zh = r1.sum; *zl = cl3;
}

// DWMulAdd_Norm — 36 flops
// Relative error bound K·7u^2.
// Computes (ah+al)*(bh+bl) + (ch+cl)*(dh+dl) with full normalization.
FLOAT_TEMPLATE_GUARD
__cuda_callable__
static constexpr __xdw_inline__ void
DWMulAdd_Norm(const T ah, const T al, const T bh, const T bl, const T ch, const T cl, const T dh, const T dl, T* __restrict__ rh, T* __restrict__ rl)
{
   // Step 1 — p = a*b via full DWTimesDW2
   T ph, pl;
   DWTimesDW2(ah, al, bh, bl, &ph, &pl);

   // Step 2 — q = c*d via full DWTimesDW2
   T qh, ql;
   DWTimesDW2(ch, cl, dh, dl, &qh, &ql);

   // Step 3 — r = p + q via maddDWPlusDW(ph, pl, qh, ql)
   maddDWPlusDW(ph, pl, qh, ql, rh, rl);
}

// DWMulAdd_Fast — 21 flops
// Relative error bound K·12u^2.
// Computes (ah+al)*(bh+bl) + (ch+cl)*(dh+dl) skipping normalization in multiplication.
FLOAT_TEMPLATE_GUARD
__cuda_callable__
static constexpr __xdw_inline__ void
DWMulAdd_Fast(const T ah, const T al, const T bh, const T bl, const T ch, const T cl, const T dh, const T dl, T* __restrict__ rh, T* __restrict__ rl)
{
   // Step 1 — p = a*b via DWTimesDWunnorm
   T ph, pl;
   DWTimesDWunnorm(ah, al, bh, bl, &ph, &pl);

   // Step 2 — q = c*d via DWTimesDWunnorm
   T qh, ql;
   DWTimesDWunnorm(ch, cl, dh, dl, &qh, &ql);

   // Step 3 — r = p + q via SloppyDWPlusDW(ph, pl, qh, ql)
   SloppyDWPlusDW(ph, pl, qh, ql, rh, rl);
}

// ComplexDWMul — computes (a + i*b) * (c + i*d) where each of a, b, c, d is a DW number.
// Inputs: (ah, al, bh, bl, ch, cl, dh, dl)
// Output: (reh, rel, imh, iml)
FLOAT_TEMPLATE_GUARD
__cuda_callable__
static constexpr __xdw_inline__ void
ComplexDWMul(const T ah, const T al, const T bh, const T bl, const T ch, const T cl, const T dh, const T dl, T* __restrict__ reh, T* __restrict__ rel, T* __restrict__ imh, T* __restrict__ iml)
{
   // Real part = ac - bd: DWMulAdd_Norm(ah, al, ch, cl, bh, bl, -dh, -dl)
   DWMulAdd_Norm(ah, al, ch, cl, bh, bl, -dh, -dl, reh, rel);
   // Imaginary part = ad + bc: DWMulAdd_Norm(ah, al, dh, dl, bh, bl, ch, cl)
   DWMulAdd_Norm(ah, al, dh, dl, bh, bl, ch, cl, imh, iml);
}

}

#endif
