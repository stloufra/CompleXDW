#ifndef DW_H
#define DW_H

#include <cmath>
#include <type_traits>

#include "src/XDWerrorFree.h"
#include "src/XDWarith.h"
#include "src/XDWTraits.h"

// Real double-word number hi + lo, |lo| <= ulp(hi)/2.
// + and - use MaddDWPlusDW, * DWTimesDW2, / DWDivDW2.
// Operators are hidden friends so a plain float/double/integer converts on either side (x + 0.5, 2 * x, x == 0.).
// T may be a SIMD vector of float/double: one independent number per lane, comparisons return the lane mask.
template< typename T >
class alignas( XDW_ARTH::XDWVector< T > ? alignof( T ) : 2 * sizeof( T ) ) DW
{

  static_assert( XDW_ARTH::XDWReal< T >, "DW<T>: T must be float, double, or a SIMD vector of them." );

  private:
  T data[ 2 ];

  public:
  using BaseType = T;
  using Lane = XDW_ARTH::lane_t< T >;

  XDW_CUDA_CALLABLE
  constexpr DW() = default;

  XDW_CUDA_CALLABLE
  constexpr DW( T hi, T lo ) : data{ hi, lo } {}

  // A wider value is split into its high and low parts, e.g. a double into DW<float> keeps 48 of its 53 bits.
  // For a vector T, the same value in every lane.
  template< typename U >
  requires std::is_arithmetic_v< U >
  XDW_CUDA_CALLABLE
  constexpr DW( U x );

  // DW<double> from DW<float> is exact; DW<float> from DW<double> rounds to 48 bits.
  template< typename U >
  requires std::floating_point< T > && std::floating_point< U > && ( !std::same_as< T, U > )
  XDW_CUDA_CALLABLE
  explicit constexpr DW( const DW< U >& x );

  // A scalar DW in every lane.
  template< typename S >
  requires XDW_ARTH::XDWVector< T > && std::same_as< S, XDW_ARTH::lane_t< T > >
  constexpr DW( const DW< S >& x ) : data{ XDW_ARTH::splat< T >( x.hi() ), XDW_ARTH::splat< T >( x.lo() ) } {}

  constexpr DW< Lane > lane( int i ) const
  requires XDW_ARTH::XDWVector< T >
  {
    return DW< Lane >( data[ 0 ][ i ], data[ 1 ][ i ] );
  }

  XDW_CUDA_CALLABLE
  constexpr T hi() const { return data[ 0 ]; }
  XDW_CUDA_CALLABLE
  constexpr T lo() const { return data[ 1 ]; }

  XDW_CUDA_CALLABLE
  constexpr T& hi() { return data[ 0 ]; }
  XDW_CUDA_CALLABLE
  constexpr T& lo() { return data[ 1 ]; }

  XDW_CUDA_CALLABLE
  constexpr T operator[]( int i ) const { return data[ i ]; }

  XDW_CUDA_CALLABLE
  constexpr T& operator[]( int i ) { return data[ i ]; }

  // Exact into a wider type (DW<float> -> double); otherwise hi, the rounded value.
  template< typename U >
  requires std::is_floating_point_v< U > && std::floating_point< T >
  XDW_CUDA_CALLABLE
  explicit constexpr operator U() const;

  XDW_CUDA_CALLABLE
  constexpr static DW< T >
  add( const DW< T >& a, const DW< T >& b );

  XDW_CUDA_CALLABLE
  constexpr static DW< T >
  sub( const DW< T >& a, const DW< T >& b );

  XDW_CUDA_CALLABLE
  constexpr static DW< T >
  mul( const DW< T >& a, const DW< T >& b );

  XDW_CUDA_CALLABLE
  constexpr static DW< T >
  div( const DW< T >& a, const DW< T >& b );

  XDW_CUDA_CALLABLE
  constexpr DW< T > operator+() const { return *this; }

  XDW_CUDA_CALLABLE
  constexpr DW< T > operator-() const { return DW< T >( -data[ 0 ], -data[ 1 ] ); }

  XDW_CUDA_CALLABLE
  constexpr DW< T >& operator+=( const DW< T >& other ) { return *this = add( *this, other ); }

  XDW_CUDA_CALLABLE
  constexpr DW< T >& operator-=( const DW< T >& other ) { return *this = sub( *this, other ); }

  XDW_CUDA_CALLABLE
  constexpr DW< T >& operator*=( const DW< T >& other ) { return *this = mul( *this, other ); }

  XDW_CUDA_CALLABLE
  constexpr DW< T >& operator/=( const DW< T >& other ) { return *this = div( *this, other ); }

  XDW_CUDA_CALLABLE
  friend constexpr XDW_INLINE DW< T > operator+( const DW< T >& a, const DW< T >& b ) { return add( a, b ); }

  XDW_CUDA_CALLABLE
  friend constexpr XDW_INLINE DW< T > operator-( const DW< T >& a, const DW< T >& b ) { return sub( a, b ); }

  XDW_CUDA_CALLABLE
  friend constexpr XDW_INLINE DW< T > operator*( const DW< T >& a, const DW< T >& b ) { return mul( a, b ); }

  XDW_CUDA_CALLABLE
  friend constexpr XDW_INLINE DW< T > operator/( const DW< T >& a, const DW< T >& b ) { return div( a, b ); }

  // Lexicographic on (hi, lo), which orders normalized values. bool for a scalar T, the lane mask for a vector T.
  XDW_CUDA_CALLABLE
  friend constexpr auto operator==( const DW< T >& a, const DW< T >& b )
  {
    return XDW_ARTH::mask_and( a.hi() == b.hi(), a.lo() == b.lo() );
  }

  XDW_CUDA_CALLABLE
  friend constexpr auto operator!=( const DW< T >& a, const DW< T >& b )
  {
    return XDW_ARTH::mask_or( a.hi() != b.hi(), a.lo() != b.lo() );
  }

  XDW_CUDA_CALLABLE
  friend constexpr auto operator<( const DW< T >& a, const DW< T >& b )
  {
    return XDW_ARTH::mask_or( a.hi() < b.hi(), XDW_ARTH::mask_and( a.hi() == b.hi(), a.lo() < b.lo() ) );
  }

  XDW_CUDA_CALLABLE
  friend constexpr auto operator<=( const DW< T >& a, const DW< T >& b )
  {
    return XDW_ARTH::mask_or( a.hi() < b.hi(), XDW_ARTH::mask_and( a.hi() == b.hi(), a.lo() <= b.lo() ) );
  }

  XDW_CUDA_CALLABLE
  friend constexpr auto operator>( const DW< T >& a, const DW< T >& b ) { return b < a; }

  XDW_CUDA_CALLABLE
  friend constexpr auto operator>=( const DW< T >& a, const DW< T >& b ) { return b <= a; }

  // a where mask is set, else b; lane by lane for a vector T.
  template< typename M >
  XDW_CUDA_CALLABLE
  friend constexpr XDW_INLINE DW< T > select( const M& mask, const DW< T >& a, const DW< T >& b )
  {
    return DW< T >( mask ? a.hi() : b.hi(), mask ? a.lo() : b.lo() );
  }

  // As std::min/std::max: a unless the other is strictly smaller/larger.
  XDW_CUDA_CALLABLE
  friend constexpr XDW_INLINE DW< T > min( const DW< T >& a, const DW< T >& b ) { return select( b < a, b, a ); }

  XDW_CUDA_CALLABLE
  friend constexpr XDW_INLINE DW< T > max( const DW< T >& a, const DW< T >& b ) { return select( a < b, b, a ); }
};

template< typename T >
template< typename U >
requires std::is_arithmetic_v< U >
XDW_CUDA_CALLABLE
constexpr DW< T >::DW( U x )
{
   using Wide = std::conditional_t< std::is_integral_v< U >, double, U >;
   const Wide w = static_cast< Wide >( x );
   const Lane hi = static_cast< Lane >( w );
   Lane lo = Lane( 0 );
   if constexpr( sizeof( Wide ) > sizeof( Lane ) )
      lo = static_cast< Lane >( w - static_cast< Wide >( hi ) );
   data[ 0 ] = XDW_ARTH::splat< T >( hi );
   data[ 1 ] = XDW_ARTH::splat< T >( lo );
}

template< typename T >
template< typename U >
requires std::floating_point< T > && std::floating_point< U > && ( !std::same_as< T, U > )
XDW_CUDA_CALLABLE
constexpr DW< T >::DW( const DW< U >& x )
{
   // Up: hi and lo may be far apart, so keep both (Fast2Sum is exact). Down: round hi, fold the rest into lo.
   T h, l;
   if constexpr( sizeof( U ) < sizeof( T ) ) {
      h = static_cast< T >( x.hi() );
      l = static_cast< T >( x.lo() );
   }
   else {
      h = static_cast< T >( x.hi() );
      l = static_cast< T >( ( x.hi() - static_cast< U >( h ) ) + x.lo() );
   }
   const XDW_ARTH::rne< T > r = XDW_ARTH::quick_two_sum( h, l );
   data[ 0 ] = r.sum;
   data[ 1 ] = r.error;
}

template< typename T >
template< typename U >
requires std::is_floating_point_v< U > && std::floating_point< T >
XDW_CUDA_CALLABLE
constexpr DW< T >::operator U() const
{
   if constexpr( sizeof( U ) > sizeof( T ) )
      return static_cast< U >( data[ 0 ] ) + static_cast< U >( data[ 1 ] );
   else
      return static_cast< U >( data[ 0 ] );
}

template< typename T >
XDW_CUDA_CALLABLE
constexpr XDW_INLINE DW< T >
DW< T >::add( const DW< T >& a, const DW< T >& b )
{
   T h, l;
   XDW_ARTH::MaddDWPlusDW( a.hi(), a.lo(), b.hi(), b.lo(), &h, &l );
   return DW< T >( h, l );
}

template< typename T >
XDW_CUDA_CALLABLE
constexpr XDW_INLINE DW< T >
DW< T >::sub( const DW< T >& a, const DW< T >& b )
{
   T h, l;
   XDW_ARTH::MaddDWPlusDW( a.hi(), a.lo(), -b.hi(), -b.lo(), &h, &l );
   return DW< T >( h, l );
}

template< typename T >
XDW_CUDA_CALLABLE
constexpr XDW_INLINE DW< T >
DW< T >::mul( const DW< T >& a, const DW< T >& b )
{
   T h, l;
   XDW_ARTH::DWTimesDW2( a.hi(), a.lo(), b.hi(), b.lo(), &h, &l );
   return DW< T >( h, l );
}

template< typename T >
XDW_CUDA_CALLABLE
constexpr XDW_INLINE DW< T >
DW< T >::div( const DW< T >& a, const DW< T >& b )
{
   T h, l;
   XDW_ARTH::DWDivDW2( a.hi(), a.lo(), b.hi(), b.lo(), &h, &l );
   return DW< T >( h, l );
}

// Functions below work lane by lane for a vector T; masks come from DW comparisons.

// As std::signbit on the value: hi carries the sign, including -0.
template< typename T >
XDW_CUDA_CALLABLE
XDW_INLINE auto
signbit( const DW< T >& x )
{
   if constexpr( XDW_ARTH::XDWVector< T > ) {
      decltype( x.hi() < x.hi() ) mask{};
      for( int i = 0; i < XDW_ARTH::lanes< T >; ++i ) mask[ i ] = std::signbit( x.hi()[ i ] ) ? -1 : 0;
      return mask;
   }
   else
      return std::signbit( x.hi() );
}

template< typename T >
XDW_CUDA_CALLABLE
XDW_INLINE DW< T >
abs( const DW< T >& x )
{
   return select( signbit( x ), -x, x );
}

// SQRTDWtoDW, relative error <= 25/8 u^2; sqrt(0) = 0.
template< typename T >
XDW_CUDA_CALLABLE
constexpr XDW_INLINE DW< T >
sqrt( const DW< T >& x )
{
   T h, l;
   XDW_ARTH::DWSqrt( x.hi(), x.lo(), &h, &l );
   return DW< T >( h, l );
}

#endif
