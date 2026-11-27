#ifndef XDW_LAZY_H
#define XDW_LAZY_H

#include "XDWarith.h"
#include "XDWTraits.h"

// Lazy sum of products for DW/XDW under NormMode::Unnormalized (XDW_ARTH::kLazyProducts).
// x*y is a ProductSum<V, 1>: a V holding the eager product, plus the factors.
// Temporary products joined by + and -, then + z, use the factors: every real product goes unnormalized
// (DWTimesDW2Unnorm) into the AddMode adds, as in DWMulAdd, and z is added with Madd as by V's +.
// A named product, or any other operation, sees the eager V, which the compiler drops when unused.

namespace XDW_ARTH::detail {

// Unnormalized pair, only fed into an add.
template< typename T >
XDW_CUDA_CALLABLE constexpr XDW_INLINE DW< T >
unnorm_product( const DW< T >& x, const DW< T >& y )
{
   T h, l;
   DWTimesDW2Unnorm( x.hi(), x.lo(), y.hi(), y.lo(), &h, &l );
   return DW< T >( h, l );
}

template< AddMode Add, typename T >
XDW_CUDA_CALLABLE constexpr XDW_INLINE DW< T >
add_unnorm_product( const DW< T >& sum, const DW< T >& x, const DW< T >& y )
{
   const DW< T > p = unnorm_product( x, y );
   T h, l;
   DWPlusDW< T, Add >( sum.hi(), sum.lo(), p.hi(), p.lo(), &h, &l );
   return DW< T >( h, l );
}

}

// x_0*y_0 + ... + x_{N-1}*y_{N-1}, V = DW<T> or XDW<T>.
template< typename V, int N >
class ProductSum : public V
{
   static_assert( XDW_ARTH::detail::DWKind< V > && N >= 1, "ProductSum<V, N>: V must be DW<T> or XDW<T>." );

   template< typename, int >
   friend class ProductSum;

   V x[ N ];
   V y[ N ];

   XDW_CUDA_CALLABLE
   constexpr ProductSum() = default;

   // Left to right, every real product unnormalized. For N == 1 a DW result is unnormalized.
   XDW_CUDA_CALLABLE
   constexpr XDW_INLINE V fused() const
   {
      using XDW_ARTH::detail::add_unnorm_product;
      using XDW_ARTH::detail::unnorm_product;
      constexpr XDW_ARTH::AddMode Add = XDW_ARTH::kAddMode;

      if constexpr( XDW_ARTH::detail::dw_kind< V >::dw ) {
         V s = unnorm_product( x[ 0 ], y[ 0 ] );
         for( int k = 1; k < N; ++k )
            s = add_unnorm_product< Add >( s, x[ k ], y[ k ] );
         return s;
      }
      else {
         // re = ac - bd, im = ad + bc, the term order of XDWmul.
         auto re = unnorm_product( real( x[ 0 ] ), real( y[ 0 ] ) );
         re = add_unnorm_product< Add >( re, imag( x[ 0 ] ), -imag( y[ 0 ] ) );
         auto im = unnorm_product( real( x[ 0 ] ), imag( y[ 0 ] ) );
         im = add_unnorm_product< Add >( im, imag( x[ 0 ] ), real( y[ 0 ] ) );
         for( int k = 1; k < N; ++k ) {
            re = add_unnorm_product< Add >( re, real( x[ k ] ), real( y[ k ] ) );
            re = add_unnorm_product< Add >( re, imag( x[ k ] ), -imag( y[ k ] ) );
            im = add_unnorm_product< Add >( im, real( x[ k ] ), imag( y[ k ] ) );
            im = add_unnorm_product< Add >( im, imag( x[ k ] ), real( y[ k ] ) );
         }
         return V( re, im );
      }
   }

   public:
   XDW_CUDA_CALLABLE
   constexpr XDW_INLINE ProductSum( const V& a, const V& b )
   requires( N == 1 )
   : V( V::mul( a, b ) ), x{ a }, y{ b } {}

   // Generic code that deduced T = ProductSum (T f(T), T acc = 0) stores a value as value * 1.
   template< typename U >
   requires( !std::is_same_v< U, ProductSum > && std::is_constructible_v< V, const U& > )
   XDW_CUDA_CALLABLE
   constexpr XDW_INLINE ProductSum( const U& value ) : V( value )
   {
      x[ 0 ] = static_cast< const V& >( *this );
      y[ 0 ] = V( 1 );
      for( int k = 1; k < N; ++k )
         x[ k ] = y[ k ] = V( 0 );
   }

   // Inherited ones would change the value and keep the old factors.
   template< typename U >
   XDW_CUDA_CALLABLE
   constexpr XDW_INLINE ProductSum& operator+=( const U& other ) { return *this = static_cast< const V& >( *this ) + other; }

   template< typename U >
   XDW_CUDA_CALLABLE
   constexpr XDW_INLINE ProductSum& operator-=( const U& other ) { return *this = static_cast< const V& >( *this ) - other; }

   template< typename U >
   XDW_CUDA_CALLABLE
   constexpr XDW_INLINE ProductSum& operator*=( const U& other ) { return *this = static_cast< const V& >( *this ) * other; }

   template< typename U >
   XDW_CUDA_CALLABLE
   constexpr XDW_INLINE ProductSum& operator/=( const U& other ) { return *this = static_cast< const V& >( *this ) / other; }

   template< int M >
   XDW_CUDA_CALLABLE
   constexpr XDW_INLINE ProductSum< V, N + M > plus( const ProductSum< V, M >& other ) const
   {
      ProductSum< V, N + M > s;
      for( int k = 0; k < N; ++k ) {
         s.x[ k ] = x[ k ];
         s.y[ k ] = y[ k ];
      }
      for( int k = 0; k < M; ++k ) {
         s.x[ N + k ] = other.x[ k ];
         s.y[ N + k ] = other.y[ k ];
      }
      static_cast< V& >( s ) = s.fused();
      return s;
   }

   XDW_CUDA_CALLABLE
   constexpr XDW_INLINE V plus( const V& addend ) const
   {
      return V::add( fused(), addend );
   }

   // Exact: one factor of each product negated.
   XDW_CUDA_CALLABLE
   constexpr XDW_INLINE ProductSum negated() const
   {
      ProductSum s = *this;
      static_cast< V& >( s ) = -static_cast< const V& >( *this );
      for( int k = 0; k < N; ++k )
         s.x[ k ] = -x[ k ];
      return s;
   }
};

// Rvalues only: a named product may have been changed through V& since its factors were stored.
template< typename V, int N, int M >
XDW_CUDA_CALLABLE
constexpr XDW_INLINE ProductSum< V, N + M >
operator+( ProductSum< V, N >&& a, ProductSum< V, M >&& b )
{
   return a.plus( b );
}

template< typename V, int N, int M >
XDW_CUDA_CALLABLE
constexpr XDW_INLINE ProductSum< V, N + M >
operator-( ProductSum< V, N >&& a, ProductSum< V, M >&& b )
{
   return a.plus( b.negated() );
}

template< typename V, int N >
XDW_CUDA_CALLABLE
constexpr XDW_INLINE V
operator+( ProductSum< V, N >&& s, const V& z )
{
   return s.plus( z );
}

template< typename V, int N >
XDW_CUDA_CALLABLE
constexpr XDW_INLINE V
operator+( const V& z, ProductSum< V, N >&& s )
{
   return s.plus( z );
}

template< typename V, int N >
XDW_CUDA_CALLABLE
constexpr XDW_INLINE V
operator-( ProductSum< V, N >&& s, const V& z )
{
   return s.plus( -z );
}

template< typename V, int N >
XDW_CUDA_CALLABLE
constexpr XDW_INLINE V
operator-( const V& z, ProductSum< V, N >&& s )
{
   return s.negated().plus( z );
}

template< typename V, int N >
XDW_CUDA_CALLABLE
constexpr XDW_INLINE V&
operator+=( V& z, ProductSum< V, N >&& s )
{
   return z = s.plus( z );
}

template< typename V, int N >
XDW_CUDA_CALLABLE
constexpr XDW_INLINE V&
operator-=( V& z, ProductSum< V, N >&& s )
{
   return z = s.negated().plus( z );
}

#endif
