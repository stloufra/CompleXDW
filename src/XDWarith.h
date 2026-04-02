#ifndef XDWARITH_H
#define XDWARITH_H

#include "XDWerrorFree.h"
#include "XDWbasicOP.h"

namespace XDW_ARTH{

// SloppyDWPlusDW — 11 flops
// Relative error <= 1
FLOAT_TEMPLATE_GUARD
__cuda_callable__
static constexpr __xdw_inline__ void
SloppyDWPlusDW(const T xh, const T xl, const T yh, const T yl, T* __restrict__ zh, T* __restrict__ zl)
{
   // TwoSum(xh, yh, &sh, &sl)
   rne<T> shl = two_sum(xh, yh);
   // v  = xl + yl
   T v = add_rn(xl, yl);
   // w  = sl + v
   T w = add_rn(shl.error, v);
   // Fast2Sum(sh, w, &zh, &zl)
   rne<T> zhl = quick_two_sum(shl.sum, w);

   *zh = zhl.sum; *zl = zhl.error;
}

// AccurateDWPlusDW — 20 flops
// Relative error < 3u^2.
FLOAT_TEMPLATE_GUARD
__cuda_callable__
static constexpr __xdw_inline__ void
AccurateDWPlusDW(const T xh, const T xl, const T yh, const T yl, T* __restrict__ zh, T* __restrict__ zl)
{
   // TwoSum(xh, yh, &sh, &sl)
   rne<T> shl = two_sum(xh, yh);
   // TwoSum(xl, yl, &th, &tl)
   rne<T> thl = two_sum(xl, yl);
   // c  = sl + th
   T c = add_rn(shl.error, thl.sum);
   // Fast2Sum(sh, c, &vh, &vl)
   rne<T> vhl = quick_two_sum(shl.sum, c);
   // w  = tl + vl
   T w = add_rn(thl.error, vhl.error);
   // Fast2Sum(vh, w, &zh, &zl)
   rne<T> zhl = quick_two_sum(vhl.sum, w);

   *zh = zhl.sum; *zl = zhl.error;
}

// maddDWPlusDW — 20 flops
// Relative error 2u^2. 
FLOAT_TEMPLATE_GUARD
__cuda_callable__
static constexpr __xdw_inline__ void
maddDWPlusDW(const T xh, const T xl, const T yh, const T yl, T* __restrict__ zh, T* __restrict__ zl)
{
   // TwoSum(xh, yh, &sh, &sl)
   rne<T> shl = two_sum(xh, yh);
   // TwoSum(xl, yl, &th, &tl)
   rne<T> thl = two_sum(xl, yl);
   // Fast2Sum(sh, th, &ah, &al)
   rne<T> ahl = quick_two_sum(shl.sum, thl.sum);
   // b  = sl + tl
   T b = add_rn(shl.error, thl.error);
   // e  = b + al
   T e = add_rn(b, ahl.error);
   // Fast2Sum(ah, e, &zh, &zl)
   rne<T> zhl = quick_two_sum(ahl.sum, e);

   *zh = zhl.sum; *zl = zhl.error;
}

// DWTimesDW2 — 8 flops
// Relative error <= 5u^2. 
FLOAT_TEMPLATE_GUARD
__cuda_callable__
static constexpr __xdw_inline__ void
DWTimesDW2(const T xh, const T xl, const T yh, const T yl, T* __restrict__ zh, T* __restrict__ zl)
{
   // TwoProdFMA(xh, yh, &ch, &cl1)
   rne<T> chl = two_prod(xh, yh);
   // tl   = xh * yl
   T tl = mul_rn(xh, yl);
   // cl2  = fma(xl, yh, tl)
   T cl2 = fma_rn(xl, yh, tl);
   // cl3  = cl1 + cl2
   T cl3 = add_rn(chl.error, cl2);
   // Fast2Sum(ch, cl3, &zh, &zl)
   rne<T> r2 = quick_two_sum(chl.sum, cl3);

   *zh = r2.sum; *zl = r2.error;
}

// DWTimesDW2Unnorm — 5 flops
// Unnormalized DW product, skips final Fast2Sum. 
// Returns (ph, pl) where pl may not satisfy |pl| <= u*|ph|.
FLOAT_TEMPLATE_GUARD
__cuda_callable__
static constexpr __xdw_inline__ void
DWTimesDW2Unnorm(const T xh, const T xl, const T yh, const T yl, T* __restrict__ zh, T* __restrict__ zl)
{
   // TwoProdFMA(xh, yh, &ch, &cl1)
   rne<T> chl = two_prod(xh, yh);
   // t   = xh * yl
   T t = mul_rn(xh, yl);
   // u   = fma(xl, yh, t)
   T u = fma_rn(xl, yh, t);
   // cl3 = cl1 + u
   T cl3 = add_rn(chl.error, u);

   *zh = chl.sum; *zl = cl3;
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

// DWMulAdd_SloppyUnnorm — 21 flops
// Relative error bound K·12u^2. (in proper region) <F4>
// Computes (ah+al)*(bh+bl) + (ch+cl)*(dh+dl) skipping normalization in multiplication.
FLOAT_TEMPLATE_GUARD
__cuda_callable__
static constexpr __xdw_inline__ void
DWMulAdd_SloppyUnnorm(const T ah, const T al, const T bh, const T bl, const T ch, const T cl, const T dh, const T dl, T* __restrict__ rh, T* __restrict__ rl)
{
   // Step 1 — p = a*b via DWTimesDW2Unnorm
   T ph, pl;
   DWTimesDW2Unnorm(ah, al, bh, bl, &ph, &pl);

   // Step 2 — q = c*d via DWTimesDW2Unnorm
   T qh, ql;
   DWTimesDW2Unnorm(ch, cl, dh, dl, &qh, &ql);

   // Step 3 — r = p + q via SloppyDWPlusDW(ph, pl, qh, ql)
   SloppyDWPlusDW(ph, pl, qh, ql, rh, rl);
}


// DWMulAdd_SloppyUnnorm — 21 flops
// Relative error bound K·12u^2. (in proper region) <F4>
// Computes (ah+al)*(bh+bl) + (ch+cl)*(dh+dl) skipping normalization in multiplication.
FLOAT_TEMPLATE_GUARD
__cuda_callable__
static constexpr __xdw_inline__ void
DWMulAdd_AccurateUnnorm(const T ah, const T al, const T bh, const T bl, const T ch, const T cl, const T dh, const T dl, T* __restrict__ rh, T* __restrict__ rl)
{
   // Step 1 — p = a*b via DWTimesDW2Unnorm
   T ph, pl;
   DWTimesDW2Unnorm(ah, al, bh, bl, &ph, &pl);

   // Step 2 — q = c*d via DWTimesDW2Unnorm
   T qh, ql;
   DWTimesDW2Unnorm(ch, cl, dh, dl, &qh, &ql);

   // Step 3 — r = p + q via SloppyDWPlusDW(ph, pl, qh, ql)
   maddDWPlusDW(ph, pl, qh, ql, rh, rl);
}

// ComplexDWMulNorm — computes (a + i*b) * (c + i*d) where each of a, b, c, d is a DW number with normalization.
// Inputs: (ah, al, bh, bl, ch, cl, dh, dl)
// Output: (reh, rel, imh, iml)
FLOAT_TEMPLATE_GUARD
__cuda_callable__
static constexpr __xdw_inline__ void
ComplexDWMulNorm(const T ah, const T al, const T bh, const T bl, const T ch, const T cl, const T dh, const T dl, T* __restrict__ reh, T* __restrict__ rel, T* __restrict__ imh, T* __restrict__ iml)
{
   // Real part = ac - bd: DWMulAdd_Norm(ah, al, ch, cl, bh, bl, -dh, -dl)
   DWMulAdd_Norm(ah, al, ch, cl, bh, bl, -dh, -dl, reh, rel);
   // Imaginary part = ad + bc: DWMulAdd_Norm(ah, al, dh, dl, bh, bl, ch, cl)
   DWMulAdd_Norm(ah, al, dh, dl, bh, bl, ch, cl, imh, iml);
}

// ComplexDWMulSloppyUnnorm — computes (a + i*b) * (c + i*d) where each of a, b, c, d is a DW number without normalization.
// Inputs: (ah, al, bh, bl, ch, cl, dh, dl)
// Output: (reh, rel, imh, iml)
FLOAT_TEMPLATE_GUARD
__cuda_callable__
static constexpr __xdw_inline__ void
ComplexDWMulSloppyUnnorm(const T ah, const T al, const T bh, const T bl, const T ch, const T cl, const T dh, const T dl, T* __restrict__ reh, T* __restrict__ rel, T* __restrict__ imh, T* __restrict__ iml)
{
   // Real part = ac - bd: DWMulAdd_SloppyUnnorm(ah, al, ch, cl, bh, bl, -dh, -dl)
   DWMulAdd_SloppyUnnorm(ah, al, ch, cl, bh, bl, -dh, -dl, reh, rel);
   // Imaginary part = ad + bc: DWMulAdd_SloppyUnnorm(ah, al, dh, dl, bh, bl, ch, cl)
   DWMulAdd_SloppyUnnorm(ah, al, dh, dl, bh, bl, ch, cl, imh, iml);
}

// ComplexDWMulAccurateUnnorm — computes (a + i*b) * (c + i*d) where each of a, b, c, d is a DW number without normalization.
// Inputs: (ah, al, bh, bl, ch, cl, dh, dl)
// Output: (reh, rel, imh, iml)
FLOAT_TEMPLATE_GUARD
__cuda_callable__
static constexpr __xdw_inline__ void
ComplexDWMulAccurateUnnorm(const T ah, const T al, const T bh, const T bl, const T ch, const T cl, const T dh, const T dl, T* __restrict__ reh, T* __restrict__ rel, T* __restrict__ imh, T* __restrict__ iml)
{
   // Real part = ac - bd: DWMulAdd_SloppyUnnorm(ah, al, ch, cl, bh, bl, -dh, -dl)
   DWMulAdd_AccurateUnnorm(ah, al, ch, cl, bh, bl, -dh, -dl, reh, rel);
   // Imaginary part = ad + bc: DWMulAdd_SloppyUnnorm(ah, al, dh, dl, bh, bl, ch, cl)
   DWMulAdd_AccurateUnnorm(ah, al, dh, dl, bh, bl, ch, cl, imh, iml);
}
}

#endif
