#ifndef XDWARITH_H
#define XDWARITH_H

#include "XDWerrorFree.h"
#include "XDWbasicOP.h"
#include "XDWTraits.h"

namespace XDW_ARTH{

//-------------------- ADD ---------------------

// SloppyDWPlusDW — 11 flops
// Relative error <= 1 (abs 7u^2)
template< std::floating_point T >
XDW_CUDA_CALLABLE
static constexpr XDW_INLINE void
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
// Relative error <= 3u^2
template< std::floating_point T >
XDW_CUDA_CALLABLE
static constexpr XDW_INLINE void
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
// Relative error <= 2u^2 
template< std::floating_point T >
XDW_CUDA_CALLABLE
static constexpr XDW_INLINE void
MaddDWPlusDW(const T xh, const T xl, const T yh, const T yl, T* __restrict__ zh, T* __restrict__ zl)
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

// Selects mode
template< std::floating_point T, AddMode Add >
XDW_CUDA_CALLABLE
static constexpr XDW_INLINE void
DWPlusDW(const T xh, const T xl, const T yh, const T yl, T* __restrict__ zh, T* __restrict__ zl)
{
   if constexpr (Add == AddMode::Madd) {
      MaddDWPlusDW(xh, xl, yh, yl, zh, zl);
   } else if constexpr (Add == AddMode::Accurate) {
      AccurateDWPlusDW(xh, xl, yh, yl, zh, zl);
   } else if constexpr (Add == AddMode::Sloppy) {
      SloppyDWPlusDW(xh, xl, yh, yl, zh, zl);
   } else {
      static_assert(Add == AddMode::Sloppy, "DWPlusDW: unhandled AddMode");
   }
}

//-------------------- COMPLEX ADD ---------------------
// (ah,al,bh,bl) op (ch,cl,dh,dl) componentwise -> (reh,rel,imh,iml)


template< std::floating_point T, AddMode Add >
XDW_CUDA_CALLABLE
static constexpr XDW_INLINE void
XDWadd(const T ah, const T al, const T bh, const T bl, const T ch, const T cl, const T dh, const T dl, T* __restrict__ reh, T* __restrict__ rel, T* __restrict__ imh, T* __restrict__ iml)
{
   DWPlusDW<T, Add>(ah, al, ch, cl, reh, rel);
   DWPlusDW<T, Add>(bh, bl, dh, dl, imh, iml);
}

template< std::floating_point T, AddMode Add >
XDW_CUDA_CALLABLE
static constexpr XDW_INLINE void
XDWsub(const T ah, const T al, const T bh, const T bl, const T ch, const T cl, const T dh, const T dl, T* __restrict__ reh, T* __restrict__ rel, T* __restrict__ imh, T* __restrict__ iml)
{
   DWPlusDW<T, Add>(ah, al, -ch, -cl, reh, rel);
   DWPlusDW<T, Add>(bh, bl, -dh, -dl, imh, iml);
}

//-------------------- MUL ---------------------

// DWTimesDW2 — 8 flops
// Relative error <= 5u^2 
template< std::floating_point T >
XDW_CUDA_CALLABLE
static constexpr XDW_INLINE void
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
// Overlap of o=3u·|zl|  
template< std::floating_point T >
XDW_CUDA_CALLABLE
static constexpr XDW_INLINE void
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

// DWTimesDW3 — 9 flops
// Relative error <= 5u^2 (4u^2, Muller & Rideau 2022)
template< std::floating_point T >
XDW_CUDA_CALLABLE
static constexpr XDW_INLINE void
DWTimesDW3(const T xh, const T xl, const T yh, const T yl, T* __restrict__ zh, T* __restrict__ zl)
{
   // TwoProdFMA(xh, yh, &ch, &cl0)
   rne<T> chl = two_prod(xh, yh);
   // t0 = xl * yl
   T t0 = mul_rn(xl, yl);
   // t1 = fma(xh, yl, t0)
   T t1 = fma_rn(xh, yl, t0);
   // t2 = fma(xl, yh, t1)
   T t2 = fma_rn(xl, yh, t1);
   // cl = cl0 + t2
   T cl = add_rn(chl.error, t2);
   // Fast2Sum(ch, cl, &zh, &zl)
   rne<T> r2 = quick_two_sum(chl.sum, cl);

   *zh = r2.sum; *zl = r2.error;
}

//-------------------- DW-FP MIXED ---------------------
// (xh,xl) op y, where y is a single (non-DW) floating-point value.
// From Valentina Popescu's thesis / CAMPARY (specAddition.h, specMultiplication.h).

// DWPlusFP — 10 flops
// Relative error <= 3u^2
template< std::floating_point T >
XDW_CUDA_CALLABLE
static constexpr XDW_INLINE void
DWPlusFP(const T xh, const T xl, const T y, T* __restrict__ zh, T* __restrict__ zl)
{
   // TwoSum(xh, y, &s, &e)
   rne<T> se = two_sum(xh, y);
   // e = xl + e
   T e = add_rn(xl, se.error);
   // Fast2Sum(s, e, &zh, &zl)
   rne<T> r2 = quick_two_sum(se.sum, e);

   *zh = r2.sum; *zl = r2.error;
}

// DWTimesFP1 — 10 flops, no FMA
// Relative error <= 2u^2
template< std::floating_point T >
XDW_CUDA_CALLABLE
static constexpr XDW_INLINE void
DWTimesFP1(const T xh, const T xl, const T y, T* __restrict__ zh, T* __restrict__ zl)
{
   // TwoProd(xh, y, &ch, &cl1)
   rne<T> chl = two_prod(xh, y);
   // t = xl * y
   T t = mul_rn(xl, y);
   // Fast2Sum(ch, t, &ch2, &e)
   rne<T> che = quick_two_sum(chl.sum, t);
   // cl2 = e + cl1
   T cl2 = add_rn(che.error, chl.error);
   // Fast2Sum(ch2, cl2, &zh, &zl)
   rne<T> r2 = quick_two_sum(che.sum, cl2);

   *zh = r2.sum; *zl = r2.error;
}

// DWTimesFP3 — 6 flops, needs FMA
// Relative error <= 2u^2
template< std::floating_point T >
XDW_CUDA_CALLABLE
static constexpr XDW_INLINE void
DWTimesFP3(const T xh, const T xl, const T y, T* __restrict__ zh, T* __restrict__ zl)
{
   // TwoProdFMA(xh, y, &ch, &cl1)
   rne<T> chl = two_prod(xh, y);
   // cl2 = fma(xl, y, cl1)
   T cl2 = fma_rn(xl, y, chl.error);
   // Fast2Sum(ch, cl2, &zh, &zl)
   rne<T> r2 = quick_two_sum(chl.sum, cl2);

   *zh = r2.sum; *zl = r2.error;
}

//-------------------- MUL ADD ---------------------
//-------------------- NORMALIZED ---------------------

// DWMulAdd_Madd_N — 36 flops
// Relative error bound K·7u^2.
template< std::floating_point T >
XDW_CUDA_CALLABLE
static constexpr XDW_INLINE void
DWMulAdd_Madd_N(const T ah, const T al, const T bh, const T bl, const T ch, const T cl, const T dh, const T dl, T* __restrict__ rh, T* __restrict__ rl)
{
   // Step 1 — p = a*b via full DWTimesDW2
   T ph, pl;
   DWTimesDW2(ah, al, bh, bl, &ph, &pl);

   // Step 2 — q = c*d via full DWTimesDW2
   T qh, ql;
   DWTimesDW2(ch, cl, dh, dl, &qh, &ql);

   // Step 3 — r = p + q via MaddDWPlusDW(ph, pl, qh, ql)
   MaddDWPlusDW(ph, pl, qh, ql, rh, rl);
}

// DWMulAdd_Accu_N — 36 flops
// Relative error bound K·8u^2.
template< std::floating_point T >
XDW_CUDA_CALLABLE
static constexpr XDW_INLINE void
DWMulAdd_Accu_N(const T ah, const T al, const T bh, const T bl, const T ch, const T cl, const T dh, const T dl, T* __restrict__ rh, T* __restrict__ rl)
{
   // Step 1 — p = a*b via full DWTimesDW2
   T ph, pl;
   DWTimesDW2(ah, al, bh, bl, &ph, &pl);

   // Step 2 — q = c*d via full DWTimesDW2
   T qh, ql;
   DWTimesDW2(ch, cl, dh, dl, &qh, &ql);

   // Step 3 — r = p + q via AccurateDWPlusDW(ph, pl, qh, ql)
   AccurateDWPlusDW(ph, pl, qh, ql, rh, rl);
}

// DWMulAdd_Slop_N — 27 flops
// Relative error bound K·8u^2. (in proper region) 
template< std::floating_point T >
XDW_CUDA_CALLABLE
static constexpr XDW_INLINE void
DWMulAdd_Slop_N(const T ah, const T al, const T bh, const T bl, const T ch, const T cl, const T dh, const T dl, T* __restrict__ rh, T* __restrict__ rl)
{
   // Step 1 — p = a*b via full DWTimesDW2
   T ph, pl;
   DWTimesDW2(ah, al, bh, bl, &ph, &pl);

   // Step 2 — q = c*d via full DWTimesDW2
   T qh, ql;
   DWTimesDW2(ch, cl, dh, dl, &qh, &ql);

   // Step 3 — r = p + q via SloppyDWPlusDW(ph, pl, qh, ql)
   SloppyDWPlusDW(ph, pl, qh, ql, rh, rl);
}

//-------------------- UNNORMAL ---------------------

// DWMulAdd_Madd_U — 30 flops
// Relative error bound K·8u^2.
template< std::floating_point T >
XDW_CUDA_CALLABLE
static constexpr XDW_INLINE void
DWMulAdd_Madd_U(const T ah, const T al, const T bh, const T bl, const T ch, const T cl, const T dh, const T dl, T* __restrict__ rh, T* __restrict__ rl)
{
   // Step 1 — p = a*b via DWTimesDW2Unnorm
   T ph, pl;
   DWTimesDW2Unnorm(ah, al, bh, bl, &ph, &pl);

   // Step 2 — q = c*d via DWTimesDW2Unnorm
   T qh, ql;
   DWTimesDW2Unnorm(ch, cl, dh, dl, &qh, &ql);

   // Step 3 — r = p + q via MaddDWPlusDW(ph, pl, qh, ql)
   MaddDWPlusDW(ph, pl, qh, ql, rh, rl);
}

// DWMulAdd_Accu_U — 30 flops
// Relative error bound K·10u^2.
template< std::floating_point T >
XDW_CUDA_CALLABLE
static constexpr XDW_INLINE void
DWMulAdd_Accu_U(const T ah, const T al, const T bh, const T bl, const T ch, const T cl, const T dh, const T dl, T* __restrict__ rh, T* __restrict__ rl)
{
   // Step 1 — p = a*b via DWTimesDW2Unnorm
   T ph, pl;
   DWTimesDW2Unnorm(ah, al, bh, bl, &ph, &pl);

   // Step 2 — q = c*d via DWTimesDW2Unnorm
   T qh, ql;
   DWTimesDW2Unnorm(ch, cl, dh, dl, &qh, &ql);

   // Step 3 — r = p + q via AccurateDWPlusDW(ph, pl, qh, ql)
   AccurateDWPlusDW(ph, pl, qh, ql, rh, rl);
}

// DWMulAdd_Slop_U — 21 flops
// Relative error bound K·12u^2. (in proper region) 
template< std::floating_point T >
XDW_CUDA_CALLABLE
static constexpr XDW_INLINE void
DWMulAdd_Slop_U(const T ah, const T al, const T bh, const T bl, const T ch, const T cl, const T dh, const T dl, T* __restrict__ rh, T* __restrict__ rl)
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

// Selects mode
template< std::floating_point T, AddMode Add >
XDW_CUDA_CALLABLE
static constexpr XDW_INLINE void
DWMulAdd(const T ah, const T al, const T bh, const T bl, const T ch, const T cl, const T dh, const T dl, T* __restrict__ rh, T* __restrict__ rl)
{
   if constexpr (Add == AddMode::Madd) {
      DWMulAdd_Madd_N(ah, al, bh, bl, ch, cl, dh, dl, rh, rl);
   } else if constexpr (Add == AddMode::Accurate) {
      DWMulAdd_Accu_N(ah, al, bh, bl, ch, cl, dh, dl, rh, rl);
   } else if constexpr (Add == AddMode::Sloppy) {
      DWMulAdd_Slop_N(ah, al, bh, bl, ch, cl, dh, dl, rh, rl);
   } else {
      static_assert(Add == AddMode::Sloppy, "DWMulAdd: unhandled AddMode");
   }
}

//-------------------- COMPLEX MUL ---------------------
// (ah,al,bh,bl) * (ch,cl,dh,dl), real = ac-bd, imag = ad+bc -> (reh,rel,imh,iml)

// Selects mode
template< std::floating_point T, AddMode Add, NormMode Norm >
XDW_CUDA_CALLABLE
static constexpr XDW_INLINE void
XDWmul(const T ah, const T al, const T bh, const T bl, const T ch, const T cl, const T dh, const T dl, T* __restrict__ reh, T* __restrict__ rel, T* __restrict__ imh, T* __restrict__ iml)
{
   if constexpr (Norm == NormMode::Normalized) {
      if constexpr (Add == AddMode::Madd) {
         DWMulAdd_Madd_N(ah, al, ch, cl, bh, bl, -dh, -dl, reh, rel);
         DWMulAdd_Madd_N(ah, al, dh, dl, bh, bl, ch, cl, imh, iml);
      } else if constexpr (Add == AddMode::Accurate) {
         DWMulAdd_Accu_N(ah, al, ch, cl, bh, bl, -dh, -dl, reh, rel);
         DWMulAdd_Accu_N(ah, al, dh, dl, bh, bl, ch, cl, imh, iml);
      } else if constexpr (Add == AddMode::Sloppy) {
         DWMulAdd_Slop_N(ah, al, ch, cl, bh, bl, -dh, -dl, reh, rel);
         DWMulAdd_Slop_N(ah, al, dh, dl, bh, bl, ch, cl, imh, iml);
      } else {
         static_assert(Add == AddMode::Sloppy, "XDWmul: unhandled AddMode");
      }
   } else if constexpr (Norm == NormMode::Unnormalized) {
      if constexpr (Add == AddMode::Madd) {
         DWMulAdd_Madd_U(ah, al, ch, cl, bh, bl, -dh, -dl, reh, rel);
         DWMulAdd_Madd_U(ah, al, dh, dl, bh, bl, ch, cl, imh, iml);
      } else if constexpr (Add == AddMode::Accurate) {
         DWMulAdd_Accu_U(ah, al, ch, cl, bh, bl, -dh, -dl, reh, rel);
         DWMulAdd_Accu_U(ah, al, dh, dl, bh, bl, ch, cl, imh, iml);
      } else if constexpr (Add == AddMode::Sloppy) {
         DWMulAdd_Slop_U(ah, al, ch, cl, bh, bl, -dh, -dl, reh, rel);
         DWMulAdd_Slop_U(ah, al, dh, dl, bh, bl, ch, cl, imh, iml);
      } else {
         static_assert(Add == AddMode::Sloppy, "XDWmul: unhandled AddMode");
      }
   } else {
      static_assert(Norm == NormMode::Unnormalized, "XDWmul: unhandled NormMode");
   }
}

}

#endif
