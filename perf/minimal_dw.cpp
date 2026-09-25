#include <cmath>
#include <complex>

#if defined( __GNUC__ ) || defined( __clang__ )
#define XDW_INLINE __attribute__( ( always_inline ) ) inline
#else
#define XDW_INLINE inline
#endif

template< typename T >
struct RNE {
    T sum;
    T error;
};

template< typename T >
XDW_INLINE T add_rn( const T x, const T y ) {
    return x + y;
}

template< typename T >
XDW_INLINE T mul_rn( const T x, const T y ) {
    return x * y;
}

template< typename T >
XDW_INLINE RNE<T> quick_two_sum( const T x, const T y ) {
    RNE<T> r;
    r.sum = x + y;
    r.error = y - (r.sum - x);
    return r;
}

template< typename T >
XDW_INLINE RNE<T> ts( const T x, const T y ) {
    RNE<T> r;
    r.sum = x + y;
    T bv = r.sum - x;
    r.error = (x - (r.sum - bv)) + (y - bv);
    return r;
}

template< typename T >
XDW_INLINE RNE<T> tp( const T x, const T y ) {
    RNE<T> r;
    r.sum = x * y;
    r.error = std::fma(x, y, -r.sum);
    return r;
}

template< typename T >
XDW_INLINE void maddDWPlusDW(const T xh, const T xl, const T yh, const T yl, T* zh, T* zl) {
    RNE<T> shl = ts(xh, yh);
    RNE<T> thl = ts(xl, yl);
    RNE<T> ahl = quick_two_sum(shl.sum, thl.sum);
    T b = add_rn(shl.error, thl.error);
    T e = add_rn(b, ahl.error);
    RNE<T> zhl = quick_two_sum(ahl.sum, e);
    *zh = zhl.sum; *zl = zhl.error;
}

template< typename T >
XDW_INLINE void SloppyDWPlusDW(const T xh, const T xl, const T yh, const T yl, T* zh, T* zl) {
    RNE<T> shl = ts(xh, yh);
    T v = add_rn(xl, yl);
    T w = add_rn(shl.error, v);
    RNE<T> zhl = quick_two_sum(shl.sum, w);
    *zh = zhl.sum; *zl = zhl.error;
}

template< typename T >
XDW_INLINE void DWTimesDW2(const T xh, const T xl, const T yh, const T yl, T* zh, T* zl) {
    RNE<T> chl = tp(xh, yh);
    T tl = mul_rn(xh, yl);
    T cl2 = std::fma(xl, yh, tl);
    T cl3 = add_rn(chl.error, cl2);
    RNE<T> r2 = quick_two_sum(chl.sum, cl3);
    *zh = r2.sum; *zl = r2.error;
}

template< typename T >
XDW_INLINE void DWTimesDW2Unnorm(const T xh, const T xl, const T yh, const T yl, T* zh, T* zl) {
    RNE<T> chl = tp(xh, yh);
    T t = mul_rn(xh, yl);
    T u = std::fma(xl, yh, t);
    T cl3 = add_rn(chl.error, u);
    *zh = chl.sum; *zl = cl3;
}

template< typename T >
XDW_INLINE void DWMulAdd_Madd_N(const T ah, const T al, const T bh, const T bl, const T ch, const T cl, const T dh, const T dl, T* rh, T* rl) {
    T ph, pl;
    DWTimesDW2(ah, al, bh, bl, &ph, &pl);
    T qh, ql;
    DWTimesDW2(ch, cl, dh, dl, &qh, &ql);
    maddDWPlusDW(ph, pl, qh, ql, rh, rl);
}

template< typename T >
XDW_INLINE void DWMulAdd_Slop_U(const T ah, const T al, const T bh, const T bl, const T ch, const T cl, const T dh, const T dl, T* rh, T* rl) {
    T ph, pl;
    DWTimesDW2Unnorm(ah, al, bh, bl, &ph, &pl);
    T qh, ql;
    DWTimesDW2Unnorm(ch, cl, dh, dl, &qh, &ql);
    SloppyDWPlusDW(ph, pl, qh, ql, rh, rl);
}

template< typename T >
XDW_INLINE void DWMulAdd_Madd_U(const T ah, const T al, const T bh, const T bl, const T ch, const T cl, const T dh, const T dl, T* rh, T* rl) {
    T ph, pl;
    DWTimesDW2Unnorm(ah, al, bh, bl, &ph, &pl);
    T qh, ql;
    DWTimesDW2Unnorm(ch, cl, dh, dl, &qh, &ql);
    maddDWPlusDW(ph, pl, qh, ql, rh, rl);
}

template< typename T >
XDW_INLINE void XDWmul_Madd_N(const T ah, const T al, const T bh, const T bl, const T ch, const T cl, const T dh, const T dl, T* reh, T* rel, T* imh, T* iml) {
    DWMulAdd_Madd_N(ah, al, ch, cl, bh, bl, -dh, -dl, reh, rel);
    DWMulAdd_Madd_N(ah, al, dh, dl, bh, bl, ch, cl, imh, iml);
}

template< typename T >
XDW_INLINE void XDWmul_Slop_U(const T ah, const T al, const T bh, const T bl, const T ch, const T cl, const T dh, const T dl, T* reh, T* rel, T* imh, T* iml) {
    DWMulAdd_Slop_U(ah, al, ch, cl, bh, bl, -dh, -dl, reh, rel);
    DWMulAdd_Slop_U(ah, al, dh, dl, bh, bl, ch, cl, imh, iml);
}

template< typename T >
XDW_INLINE void XDWmul_Madd_U(const T ah, const T al, const T bh, const T bl, const T ch, const T cl, const T dh, const T dl, T* reh, T* rel, T* imh, T* iml) {
    DWMulAdd_Madd_U(ah, al, ch, cl, bh, bl, -dh, -dl, reh, rel);
    DWMulAdd_Madd_U(ah, al, dh, dl, bh, bl, ch, cl, imh, iml);
}

enum class AddMode { Madd, Sloppy };
enum class NormMode { Normalized, Unnormalized };

template< typename T >
struct alignas( 4 * sizeof( T ) ) XDW {
    T re_h, re_l, im_h, im_l;
    
    constexpr XDW() = default;
    constexpr XDW( T re ) : re_h(re), re_l(T(0)), im_h(T(0)), im_l(T(0)) {}
    constexpr XDW( T re, T im ) : re_h(re), re_l(T(0)), im_h(im), im_l(T(0)) {}
    constexpr XDW( T re_h_, T re_l_, T im_h_, T im_l_ ) : re_h(re_h_), re_l(re_l_), im_h(im_h_), im_l(im_l_) {}
};

template< typename T, AddMode Add, NormMode Norm >
XDW_INLINE XDW<T> mul( const XDW<T>& a, const XDW<T>& b ) {
    XDW<T> r;
    if constexpr (Add == AddMode::Madd && Norm == NormMode::Normalized) {
        XDWmul_Madd_N(a.re_h, a.re_l, a.im_h, a.im_l, b.re_h, b.re_l, b.im_h, b.im_l, &r.re_h, &r.re_l, &r.im_h, &r.im_l);
    } else if constexpr (Add == AddMode::Madd && Norm == NormMode::Unnormalized) {
        XDWmul_Madd_U(a.re_h, a.re_l, a.im_h, a.im_l, b.re_h, b.re_l, b.im_h, b.im_l, &r.re_h, &r.re_l, &r.im_h, &r.im_l);
    } else if constexpr (Add == AddMode::Sloppy && Norm == NormMode::Unnormalized) {
        XDWmul_Slop_U(a.re_h, a.re_l, a.im_h, a.im_l, b.re_h, b.re_l, b.im_h, b.im_l, &r.re_h, &r.re_l, &r.im_h, &r.im_l);
    } else {
        static_assert(Add == AddMode::Sloppy && Norm == NormMode::Unnormalized, "mul: unhandled (AddMode, NormMode)");
    }
    return r;
}

template< typename T >
XDW_INLINE XDW<T> operator+( const XDW<T>& a, const XDW<T>& b ) {
    T reh, rel, imh, iml;
    maddDWPlusDW(a.re_h, a.re_l, b.re_h, b.re_l, &reh, &rel);
    maddDWPlusDW(a.im_h, a.im_l, b.im_h, b.im_l, &imh, &iml);
    return XDW<T>(reh, rel, imh, iml);
}

template< typename T >
XDW_INLINE XDW<T> operator*( const XDW<T>& a, const XDW<T>& b ) {
    return mul<T, AddMode::Madd, NormMode::Normalized>(a, b);
}

using ComplexD = XDW<double>;

int main() {
    ComplexD a(1.5, 0.1, 2.3, 0.01);
    ComplexD b(0.5, 0.02, 1.1, 0.005);

    auto c = a * b;

    volatile double sink = c.re_h;
    (void)sink;

    auto d = mul<double, AddMode::Sloppy, NormMode::Unnormalized>(a, b);
    volatile double sink2 = d.re_h;
    (void)sink2;

    return 0;
}