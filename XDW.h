#ifndef XDW_H
#define XDW_H

#include "src/XDWerrorFree.h"
#include "src/XDWarith.h"
#include "src/XDWTraits.h"
#include "DW.h"
#include <complex>
#include <type_traits>

// T may be a SIMD vector of float/double: one independent complex number per lane.
template< typename T >
class alignas( XDW_ARTH::XDWVector< T > ? alignof( T ) : 4 * sizeof( T ) ) XDW
{

  static_assert( XDW_ARTH::XDWReal< T >, "XDW<T>: T must be float, double, or a SIMD vector of them." );

  private:
  T data[ 4 ];

  public:
  using BaseType = T;

  XDW_CUDA_CALLABLE
  constexpr XDW() = default;

  XDW_CUDA_CALLABLE
  constexpr XDW( const XDW& other ) = default;

  XDW_CUDA_CALLABLE
  constexpr XDW( XDW&& other ) noexcept = default;

  XDW_CUDA_CALLABLE
  constexpr XDW& operator=( const XDW& other ) = default;

  XDW_CUDA_CALLABLE
  constexpr XDW& operator=( XDW&& other ) noexcept = default;

  // Split like a plain number: std::complex<double> into XDW<float> keeps the low parts.
  template< typename U >
  requires std::is_arithmetic_v< U >
  XDW_CUDA_CALLABLE
  constexpr XDW( const std::complex< U >& c ) : XDW( DW< T >( c.real() ), DW< T >( c.imag() ) ) {}

  // Exact into a wider type (XDW<float> -> std::complex<double>); otherwise the high words.
  template< typename U >
  requires std::is_floating_point_v< U > && std::floating_point< T >
  explicit constexpr operator std::complex< U >() const
  {
    return std::complex< U >( static_cast< U >( DW< T >( data[ 0 ], data[ 1 ] ) ),
                              static_cast< U >( DW< T >( data[ 2 ], data[ 3 ] ) ) );
  }

  // Plain numbers go through DW, so a double into XDW<float> keeps its low part.
  template< typename U >
  requires std::is_arithmetic_v< U >
  XDW_CUDA_CALLABLE
  constexpr XDW( U re ) : XDW( DW< T >( re ), DW< T >( T{}, T{} ) ) {}

  template< typename U, typename V >
  requires std::is_arithmetic_v< U > && std::is_arithmetic_v< V >
  XDW_CUDA_CALLABLE
  constexpr XDW( U re, V im ) : XDW( DW< T >( re ), DW< T >( im ) ) {}

  XDW_CUDA_CALLABLE
  constexpr XDW( const DW< T >& re, const DW< T >& im = DW< T >( T{}, T{} ) )
  : data{ re.hi(), re.lo(), im.hi(), im.lo() } {}

  // A scalar XDW in every lane.
  template< typename S >
  requires XDW_ARTH::XDWVector< T > && std::same_as< S, XDW_ARTH::lane_t< T > >
  constexpr XDW( const XDW< S >& z )
  : data{ XDW_ARTH::splat< T >( z.re_h() ), XDW_ARTH::splat< T >( z.re_l() ),
          XDW_ARTH::splat< T >( z.im_h() ), XDW_ARTH::splat< T >( z.im_l() ) } {}

  constexpr XDW< XDW_ARTH::lane_t< T > > lane( int i ) const
  requires XDW_ARTH::XDWVector< T >
  {
    return XDW< XDW_ARTH::lane_t< T > >( data[ 0 ][ i ], data[ 1 ][ i ], data[ 2 ][ i ], data[ 3 ][ i ] );
  }

  XDW_CUDA_CALLABLE
  constexpr XDW( T re_h, T re_l, T im_h, T im_l );

  XDW_CUDA_CALLABLE
  constexpr T re_h() const { return data[ 0 ]; }
  XDW_CUDA_CALLABLE
  constexpr T re_l() const { return data[ 1 ]; }
  XDW_CUDA_CALLABLE
  constexpr T im_h() const { return data[ 2 ]; }
  XDW_CUDA_CALLABLE
  constexpr T im_l() const { return data[ 3 ]; }

  XDW_CUDA_CALLABLE
  constexpr T& re_h() { return data[ 0 ]; }
  XDW_CUDA_CALLABLE
  constexpr T& re_l() { return data[ 1 ]; }
  XDW_CUDA_CALLABLE
  constexpr T& im_h() { return data[ 2 ]; }
  XDW_CUDA_CALLABLE
  constexpr T& im_l() { return data[ 3 ]; }

  XDW_CUDA_CALLABLE
  constexpr T operator[]( int i ) const { return data[ i ]; }

  XDW_CUDA_CALLABLE
  constexpr T& operator[]( int i ) { return data[ i ]; }

  XDW_CUDA_CALLABLE
  constexpr XDW< T >
  operator+() const;

  XDW_CUDA_CALLABLE
  constexpr XDW< T >
  operator-() const;

  XDW_CUDA_CALLABLE
  constexpr XDW< T >&
  operator+=( const XDW< T >& other );

  XDW_CUDA_CALLABLE
  constexpr XDW< T >&
  operator-=( const XDW< T >& other );

  XDW_CUDA_CALLABLE
  constexpr XDW< T >&
  operator*=( const XDW< T >& other );

  XDW_CUDA_CALLABLE
  constexpr XDW< T >&
  operator/=( const XDW< T >& other );

  // a where mask is set, else b; lane by lane for a vector T. A friend, so a scalar XDW or a plain number converts.
  template< typename M >
  XDW_CUDA_CALLABLE
  friend constexpr XDW_INLINE XDW< T > select( const M& mask, const XDW< T >& a, const XDW< T >& b )
  {
    return XDW< T >( mask ? a.re_h() : b.re_h(), mask ? a.re_l() : b.re_l(),
                     mask ? a.im_h() : b.im_h(), mask ? a.im_l() : b.im_l() );
  }

  XDW_CUDA_CALLABLE
  constexpr static XDW< T >
  add( const XDW< T >& a, const XDW< T >& b );

  XDW_CUDA_CALLABLE
  constexpr static XDW< T >
  sub( const XDW< T >& a, const XDW< T >& b );

  template< XDW_ARTH::AddMode Add = XDW_ARTH::kAddMode, XDW_ARTH::NormMode Norm = XDW_ARTH::kNormMode >
  XDW_CUDA_CALLABLE
  constexpr static XDW< T >
  mul( const XDW< T >& a, const XDW< T >& b );

  template< XDW_ARTH::DivMode Div = XDW_ARTH::kDivMode, XDW_ARTH::AddMode Add = XDW_ARTH::kAddMode, XDW_ARTH::NormMode Norm = XDW_ARTH::kNormMode >
  XDW_CUDA_CALLABLE
  constexpr static XDW< T >
  div( const XDW< T >& a, const XDW< T >& b );
};

template< typename T >
XDW_CUDA_CALLABLE
constexpr XDW< T >::XDW( T re_h_, T re_l_, T im_h_, T im_l_ )
   : data{ re_h_, re_l_, im_h_, im_l_ }
{}

template< typename T >
XDW_CUDA_CALLABLE
constexpr XDW< T >
XDW< T >::operator+() const
{
   return *this;
}

template< typename T >
XDW_CUDA_CALLABLE
constexpr XDW< T >
XDW< T >::operator-() const
{
   return XDW< T >( -data[ 0 ], -data[ 1 ], -data[ 2 ], -data[ 3 ] );
}

template< typename T >
XDW_CUDA_CALLABLE
constexpr XDW_INLINE XDW< T >
XDW< T >::add( const XDW< T >& a, const XDW< T >& b )
{
   T reh, rel, imh, iml;
   XDW_ARTH::XDWadd< T, XDW_ARTH::AddMode::Madd >( a.re_h(), a.re_l(), a.im_h(), a.im_l(),
                                b.re_h(), b.re_l(), b.im_h(), b.im_l(),
                                &reh, &rel, &imh, &iml );
   return XDW< T >( reh, rel, imh, iml );
}

template< typename T >
XDW_CUDA_CALLABLE
constexpr XDW_INLINE XDW< T >
XDW< T >::sub( const XDW< T >& a, const XDW< T >& b )
{
   T reh, rel, imh, iml;
   XDW_ARTH::XDWsub< T, XDW_ARTH::AddMode::Madd >( a.re_h(), a.re_l(), a.im_h(), a.im_l(),
                                b.re_h(), b.re_l(), b.im_h(), b.im_l(),
                                &reh, &rel, &imh, &iml );
   return XDW< T >( reh, rel, imh, iml );
}

template< typename T >
template< XDW_ARTH::AddMode Add, XDW_ARTH::NormMode Norm >
XDW_CUDA_CALLABLE
constexpr XDW_INLINE XDW< T >
XDW< T >::mul( const XDW< T >& a, const XDW< T >& b )
{
   T reh, rel, imh, iml;
   XDW_ARTH::XDWmul< T, Add, Norm >( a.re_h(), a.re_l(), a.im_h(), a.im_l(),
                                      b.re_h(), b.re_l(), b.im_h(), b.im_l(), &reh, &rel, &imh, &iml );
   return XDW< T >( reh, rel, imh, iml );
}

template< typename T >
template< XDW_ARTH::DivMode Div, XDW_ARTH::AddMode Add, XDW_ARTH::NormMode Norm >
XDW_CUDA_CALLABLE
constexpr XDW_INLINE XDW< T >
XDW< T >::div( const XDW< T >& a, const XDW< T >& b )
{
   T reh, rel, imh, iml;
   XDW_ARTH::XDWDiv< T, Div, Add, Norm >( a.re_h(), a.re_l(), a.im_h(), a.im_l(),
                                           b.re_h(), b.re_l(), b.im_h(), b.im_l(), &reh, &rel, &imh, &iml );
   return XDW< T >( reh, rel, imh, iml );
}



template< typename T >
XDW_CUDA_CALLABLE
constexpr XDW_INLINE XDW< T >&
XDW< T >::operator+=( const XDW< T >& other )
{
   return *this = *this + other;
}

template< typename T >
XDW_CUDA_CALLABLE
constexpr XDW_INLINE XDW< T >&
XDW< T >::operator-=( const XDW< T >& other )
{
   return *this = *this - other;
}

template< typename T >
XDW_CUDA_CALLABLE
constexpr XDW_INLINE XDW< T >&
XDW< T >::operator*=( const XDW< T >& other )
{
   return *this = *this * other;
}

template< typename T >
XDW_CUDA_CALLABLE
constexpr XDW_INLINE XDW< T >&
XDW< T >::operator/=( const XDW< T >& other )
{
   return *this = *this / other;
}

template< typename T >
XDW_CUDA_CALLABLE
constexpr XDW_INLINE XDW< T >
operator+( const XDW< T >& a, const XDW< T >& b )
{
   return XDW< T >::add( a, b );
}

template< typename T >
XDW_CUDA_CALLABLE
constexpr XDW_INLINE XDW< T >
operator-( const XDW< T >& a, const XDW< T >& b )
{
   return XDW< T >::sub( a, b );
}

template< typename T >
XDW_CUDA_CALLABLE
constexpr XDW_INLINE XDW< T >
operator*( const XDW< T >& a, const XDW< T >& b )
{
   return XDW< T >::mul( a, b );
}

template< typename T >
XDW_CUDA_CALLABLE
constexpr XDW_INLINE XDW< T >
operator/( const XDW< T >& a, const XDW< T >& b )
{
   return XDW< T >::div( a, b );
}

// bool for a scalar T, the lane mask for a vector T.
template< typename T >
XDW_CUDA_CALLABLE
constexpr XDW_INLINE auto
operator==( const XDW< T >& a, const XDW< T >& b )
{
   using XDW_ARTH::mask_and;
   return mask_and( mask_and( a.re_h() == b.re_h(), a.re_l() == b.re_l() ),
                    mask_and( a.im_h() == b.im_h(), a.im_l() == b.im_l() ) );
}

template< typename T >
XDW_CUDA_CALLABLE
constexpr XDW_INLINE XDW< T >
conj( const XDW< T >& z )
{
   return XDW< T >( z.re_h(), z.re_l(), -z.im_h(), -z.im_l() );
}

template< typename T >
XDW_CUDA_CALLABLE
constexpr XDW_INLINE DW< T >
real( const XDW< T >& z )
{
   return DW< T >( z.re_h(), z.re_l() );
}

template< typename T >
XDW_CUDA_CALLABLE
constexpr XDW_INLINE DW< T >
imag( const XDW< T >& z )
{
   return DW< T >( z.im_h(), z.im_l() );
}

// |z|^2 = z * z', no sqrt.
template< typename T, XDW_ARTH::AddMode Add = XDW_ARTH::kAddMode, XDW_ARTH::NormMode Norm = XDW_ARTH::kNormMode >
XDW_CUDA_CALLABLE
constexpr XDW_INLINE DW< T >
norm( const XDW< T >& z )
{
   T rh, rl;
   XDW_ARTH::DWPowAdd< T, Add, Norm >( z.re_h(), z.re_l(), z.im_h(), z.im_l(), &rh, &rl );
   return DW< T >( rh, rl );
}

//-------------------- XDW <-> DW ---------------------
// A real operand only touches the parts it multiplies or adds: componentwise DW + - * /.
// The DW side is non-deduced, so a plain number also works (z * 0.5, 2 * z).

template< typename T >
XDW_CUDA_CALLABLE
constexpr XDW_INLINE XDW< T >
operator+( const XDW< T >& z, const std::type_identity_t< DW< T > >& x )
{
   return XDW< T >( real( z ) + x, imag( z ) );
}

template< typename T >
XDW_CUDA_CALLABLE
constexpr XDW_INLINE XDW< T >
operator+( const std::type_identity_t< DW< T > >& x, const XDW< T >& z )
{
   return z + x;
}

template< typename T >
XDW_CUDA_CALLABLE
constexpr XDW_INLINE XDW< T >
operator-( const XDW< T >& z, const std::type_identity_t< DW< T > >& x )
{
   return XDW< T >( real( z ) - x, imag( z ) );
}

template< typename T >
XDW_CUDA_CALLABLE
constexpr XDW_INLINE XDW< T >
operator-( const std::type_identity_t< DW< T > >& x, const XDW< T >& z )
{
   return XDW< T >( x - real( z ), -imag( z ) );
}

template< typename T >
XDW_CUDA_CALLABLE
constexpr XDW_INLINE XDW< T >
operator*( const XDW< T >& z, const std::type_identity_t< DW< T > >& x )
{
   return XDW< T >( real( z ) * x, imag( z ) * x );
}

template< typename T >
XDW_CUDA_CALLABLE
constexpr XDW_INLINE XDW< T >
operator*( const std::type_identity_t< DW< T > >& x, const XDW< T >& z )
{
   return z * x;
}

template< typename T >
XDW_CUDA_CALLABLE
constexpr XDW_INLINE XDW< T >
operator/( const XDW< T >& z, const std::type_identity_t< DW< T > >& x )
{
   return XDW< T >( real( z ) / x, imag( z ) / x );
}

// A real over a complex is a full complex division.
template< typename T >
XDW_CUDA_CALLABLE
constexpr XDW_INLINE XDW< T >
operator/( const std::type_identity_t< DW< T > >& x, const XDW< T >& z )
{
   return XDW< T >( x ) / z;
}

//-------------------- MIXED OPERANDS ---------------------
// An XDW meeting a plain number, or a DW/XDW of another base: a scalar DW/XDW next to a vector one
// of the same lane type, e.g. XDW<float> * XDW<float x8>. The operands are converted to the common
// base (plain numbers split, scalars put in every lane) and the same-base operator above runs.
// Two DWs of different bases are handled by DW's own operators.

namespace XDW_ARTH::detail {

template< typename A >
struct dw_kind { static constexpr bool dw = false, xdw = false; };

template< typename T >
struct dw_kind< DW< T > > { static constexpr bool dw = true, xdw = false; using base = T; };

template< typename T >
struct dw_kind< XDW< T > > { static constexpr bool dw = false, xdw = true; using base = T; };

template< typename A >
concept DWKind = dw_kind< A >::dw || dw_kind< A >::xdw;

// The base both operands convert to; void if there is none.
template< typename A, typename B >
struct common_base { using type = void; };

template< DWKind A, typename B >
requires std::is_arithmetic_v< B >
struct common_base< A, B > { using type = typename dw_kind< A >::base; };

template< typename A, DWKind B >
requires std::is_arithmetic_v< A >
struct common_base< A, B > { using type = typename dw_kind< B >::base; };

template< typename S, typename V >
constexpr bool lane_of = XDWVector< V > && std::same_as< lane_t< V >, S >;

template< DWKind A, DWKind B >
struct common_base< A, B >
{
   using TA = typename dw_kind< A >::base;
   using TB = typename dw_kind< B >::base;
   using type = std::conditional_t< lane_of< TB, TA >, TA, std::conditional_t< lane_of< TA, TB >, TB, void > >;
};

template< typename A, typename B >
concept MixedXDW = ( dw_kind< A >::xdw || dw_kind< B >::xdw ) && !std::is_void_v< typename common_base< A, B >::type >;

template< typename T, typename A >
XDW_CUDA_CALLABLE constexpr auto lift( const A& a )
{
   if constexpr( dw_kind< A >::xdw )
      return XDW< T >( a );
   else
      return DW< T >( a );
}

}

template< typename A, typename B >
requires XDW_ARTH::detail::MixedXDW< A, B >
XDW_CUDA_CALLABLE
constexpr XDW_INLINE auto
operator+( const A& a, const B& b )
{
   using T = typename XDW_ARTH::detail::common_base< A, B >::type;
   return XDW_ARTH::detail::lift< T >( a ) + XDW_ARTH::detail::lift< T >( b );
}

template< typename A, typename B >
requires XDW_ARTH::detail::MixedXDW< A, B >
XDW_CUDA_CALLABLE
constexpr XDW_INLINE auto
operator-( const A& a, const B& b )
{
   using T = typename XDW_ARTH::detail::common_base< A, B >::type;
   return XDW_ARTH::detail::lift< T >( a ) - XDW_ARTH::detail::lift< T >( b );
}

template< typename A, typename B >
requires XDW_ARTH::detail::MixedXDW< A, B >
XDW_CUDA_CALLABLE
constexpr XDW_INLINE auto
operator*( const A& a, const B& b )
{
   using T = typename XDW_ARTH::detail::common_base< A, B >::type;
   return XDW_ARTH::detail::lift< T >( a ) * XDW_ARTH::detail::lift< T >( b );
}

template< typename A, typename B >
requires XDW_ARTH::detail::MixedXDW< A, B >
XDW_CUDA_CALLABLE
constexpr XDW_INLINE auto
operator/( const A& a, const B& b )
{
   using T = typename XDW_ARTH::detail::common_base< A, B >::type;
   return XDW_ARTH::detail::lift< T >( a ) / XDW_ARTH::detail::lift< T >( b );
}

#endif
