#ifndef DW_H
#define DW_H

#include <type_traits>

#include "src/XDWerrorFree.h"
#include "src/XDWarith.h"
#include "src/XDWTraits.h"

// Real double-word number hi + lo, |lo| <= ulp(hi)/2.
// + and - use MaddDWPlusDW, * DWTimesDW2, / DWDivDW2.
// Operators are hidden friends so a plain float/double/integer converts on either side (x + 0.5, 2 * x, x == 0.).
template< typename T >
class alignas( 2 * sizeof( T ) ) DW
{

  static_assert( std::is_same_v< T, float > || std::is_same_v< T, double >
                 ,"DW<T> can only be instantiated with float or double." );

  private:
  T data[ 2 ];

  public:
  using BaseType = T;

  XDW_CUDA_CALLABLE
  constexpr DW() = default;

  XDW_CUDA_CALLABLE
  constexpr DW( T hi, T lo ) : data{ hi, lo } {}

  // A wider value is split into its high and low parts, e.g. a double into DW<float> keeps 48 of its 53 bits.
  template< typename U >
  requires std::is_arithmetic_v< U >
  XDW_CUDA_CALLABLE
  constexpr DW( U x );

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
  requires std::is_floating_point_v< U >
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

  // Lexicographic on (hi, lo), which orders normalized values.
  XDW_CUDA_CALLABLE
  friend constexpr bool operator==( const DW< T >& a, const DW< T >& b ) { return a.hi() == b.hi() && a.lo() == b.lo(); }

  XDW_CUDA_CALLABLE
  friend constexpr bool operator!=( const DW< T >& a, const DW< T >& b ) { return !( a == b ); }

  XDW_CUDA_CALLABLE
  friend constexpr bool operator<( const DW< T >& a, const DW< T >& b ) { return a.hi() < b.hi() || ( a.hi() == b.hi() && a.lo() < b.lo() ); }

  XDW_CUDA_CALLABLE
  friend constexpr bool operator>( const DW< T >& a, const DW< T >& b ) { return b < a; }

  XDW_CUDA_CALLABLE
  friend constexpr bool operator<=( const DW< T >& a, const DW< T >& b ) { return !( b < a ); }

  XDW_CUDA_CALLABLE
  friend constexpr bool operator>=( const DW< T >& a, const DW< T >& b ) { return !( a < b ); }
};

template< typename T >
template< typename U >
requires std::is_arithmetic_v< U >
XDW_CUDA_CALLABLE
constexpr DW< T >::DW( U x )
{
   using Wide = std::conditional_t< std::is_integral_v< U >, double, U >;
   const Wide w = static_cast< Wide >( x );
   data[ 0 ] = static_cast< T >( w );
   if constexpr( sizeof( Wide ) > sizeof( T ) )
      data[ 1 ] = static_cast< T >( w - static_cast< Wide >( data[ 0 ] ) );
   else
      data[ 1 ] = T( 0 );
}

template< typename T >
template< typename U >
requires std::is_floating_point_v< U >
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

#endif
