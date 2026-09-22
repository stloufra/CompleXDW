#ifndef COMPLEX_DOUBLE_H
#define COMPLEX_DOUBLE_H

#include "src/XDWerrorFree.h"
#include "src/XDWarith.h"
#include "src/XDWTraits.h"
#include <complex>

template< typename T >
class alignas( 4 * sizeof( T ) ) ComplexDouble
{

  static_assert( std::is_same_v< T, float > || std::is_same_v< T, double >
                 ,"ComplexDouble<T> can only be instantiated with float or double." );

  private:
  T data[ 4 ];

  public:
  using BaseType = T;

  __cuda_callable__
  constexpr ComplexDouble() = default;

  __cuda_callable__
  constexpr ComplexDouble( const ComplexDouble& other ) = default;

  __cuda_callable__
  constexpr ComplexDouble( ComplexDouble&& other ) noexcept = default;

  __cuda_callable__
  constexpr ComplexDouble& operator=( const ComplexDouble& other ) = default;

  __cuda_callable__
  constexpr ComplexDouble& operator=( ComplexDouble&& other ) noexcept = default;

  __cuda_callable__
  constexpr ComplexDouble( const std::complex< T >& c );

  __cuda_callable__
  constexpr ComplexDouble( T re );

  __cuda_callable__
  constexpr ComplexDouble( T re, T im );

  __cuda_callable__
  constexpr ComplexDouble( T re_h, T re_l, T im_h, T im_l );

  __cuda_callable__
  constexpr T re_h() const { return data[ 0 ]; }
  __cuda_callable__
  constexpr T re_l() const { return data[ 1 ]; }
  __cuda_callable__
  constexpr T im_h() const { return data[ 2 ]; }
  __cuda_callable__
  constexpr T im_l() const { return data[ 3 ]; }

  __cuda_callable__
  constexpr T& re_h() { return data[ 0 ]; }
  __cuda_callable__
  constexpr T& re_l() { return data[ 1 ]; }
  __cuda_callable__
  constexpr T& im_h() { return data[ 2 ]; }
  __cuda_callable__
  constexpr T& im_l() { return data[ 3 ]; }

  __cuda_callable__
  constexpr T operator[]( int i ) const { return data[ i ]; }

  __cuda_callable__
  constexpr T& operator[]( int i ) { return data[ i ]; }

  __cuda_callable__
  constexpr ComplexDouble< T >
  operator+() const;

  __cuda_callable__
  constexpr ComplexDouble< T >
  operator-() const;

  __cuda_callable__
  constexpr ComplexDouble< T >&
  operator+=( const ComplexDouble< T >& other );

  __cuda_callable__
  constexpr ComplexDouble< T >&
  operator-=( const ComplexDouble< T >& other );

  __cuda_callable__
  constexpr ComplexDouble< T >&
  operator*=( const ComplexDouble< T >& other );

  template< XDW_ARTH::AddMode Add = XDW_ARTH::kAddMode >
  __cuda_callable__
  constexpr static ComplexDouble< T >
  add( const ComplexDouble< T >& a, const ComplexDouble< T >& b );

  template< XDW_ARTH::AddMode Add = XDW_ARTH::kAddMode >
  __cuda_callable__
  constexpr static ComplexDouble< T >
  sub( const ComplexDouble< T >& a, const ComplexDouble< T >& b );

  template< XDW_ARTH::AddMode Add = XDW_ARTH::kAddMode, XDW_ARTH::NormMode Norm = XDW_ARTH::kNormMode >
  __cuda_callable__
  constexpr static ComplexDouble< T >
  mul( const ComplexDouble< T >& a, const ComplexDouble< T >& b );
};

template< typename T >
__cuda_callable__
constexpr ComplexDouble< T >::ComplexDouble( T re_h_, T re_l_, T im_h_, T im_l_ )
   : data{ re_h_, re_l_, im_h_, im_l_ }
{}

template< typename T >
__cuda_callable__
constexpr ComplexDouble< T >::ComplexDouble( const std::complex< T >& c )
{
   data[ 0 ] = c.real();
   data[ 1 ] = static_cast< T >(0.0F);
   data[ 2 ] = c.imag();
   data[ 3 ] = static_cast< T >(0.0F);
}

template< typename T >
__cuda_callable__
constexpr ComplexDouble< T >::ComplexDouble( T re_ )
   : data{ re_, T(0), T(0), T(0) }
{}

template< typename T >
__cuda_callable__
constexpr ComplexDouble< T >::ComplexDouble( T re_, T im_)
   : data{ re_, T(0), im_, T(0) }
{}

template< typename T >
__cuda_callable__
constexpr ComplexDouble< T >
ComplexDouble< T >::operator+() const
{
   return *this;
}

template< typename T >
__cuda_callable__
constexpr ComplexDouble< T >
ComplexDouble< T >::operator-() const
{
   return ComplexDouble< T >( -data[ 0 ], -data[ 1 ], -data[ 2 ], -data[ 3 ] );
}

template< typename T >
template< XDW_ARTH::AddMode Add >
__cuda_callable__
constexpr __xdw_inline__ ComplexDouble< T >
ComplexDouble< T >::add( const ComplexDouble< T >& a, const ComplexDouble< T >& b )
{
   T reh, rel, imh, iml;
   XDW_ARTH::XDWadd< T, Add >( a.re_h(), a.re_l(), a.im_h(), a.im_l(),
                                b.re_h(), b.re_l(), b.im_h(), b.im_l(),
                                &reh, &rel, &imh, &iml );
   return ComplexDouble< T >( reh, rel, imh, iml );
}

template< typename T >
template< XDW_ARTH::AddMode Add >
__cuda_callable__
constexpr __xdw_inline__ ComplexDouble< T >
ComplexDouble< T >::sub( const ComplexDouble< T >& a, const ComplexDouble< T >& b )
{
   T reh, rel, imh, iml;
   XDW_ARTH::XDWsub< T, Add >( a.re_h(), a.re_l(), a.im_h(), a.im_l(),
                                b.re_h(), b.re_l(), b.im_h(), b.im_l(),
                                &reh, &rel, &imh, &iml );
   return ComplexDouble< T >( reh, rel, imh, iml );
}

template< typename T >
template< XDW_ARTH::AddMode Add, XDW_ARTH::NormMode Norm >
__cuda_callable__
constexpr __xdw_inline__ ComplexDouble< T >
ComplexDouble< T >::mul( const ComplexDouble< T >& a, const ComplexDouble< T >& b )
{
   T reh, rel, imh, iml;
   XDW_ARTH::XDWmul< T, Add, Norm >( a.re_h(), a.re_l(), a.im_h(), a.im_l(),
                                      b.re_h(), b.re_l(), b.im_h(), b.im_l(), &reh, &rel, &imh, &iml );
   return ComplexDouble< T >( reh, rel, imh, iml );
}



template< typename T >
__cuda_callable__
constexpr __xdw_inline__ ComplexDouble< T >&
ComplexDouble< T >::operator+=( const ComplexDouble< T >& other )
{
   return *this = *this + other;
}

template< typename T >
__cuda_callable__
constexpr __xdw_inline__ ComplexDouble< T >&
ComplexDouble< T >::operator-=( const ComplexDouble< T >& other )
{
   return *this = *this - other;
}

template< typename T >
__cuda_callable__
constexpr __xdw_inline__ ComplexDouble< T >&
ComplexDouble< T >::operator*=( const ComplexDouble< T >& other )
{
   return *this = *this * other;
}

template< typename T >
__cuda_callable__
constexpr __xdw_inline__ ComplexDouble< T >
operator+( const ComplexDouble< T >& a, const ComplexDouble< T >& b )
{
   return ComplexDouble< T >::add( a, b );
}

template< typename T >
__cuda_callable__
constexpr __xdw_inline__ ComplexDouble< T >
operator-( const ComplexDouble< T >& a, const ComplexDouble< T >& b )
{
   return ComplexDouble< T >::sub( a, b );
}

template< typename T >
__cuda_callable__
constexpr __xdw_inline__ ComplexDouble< T >
operator*( const ComplexDouble< T >& a, const ComplexDouble< T >& b )
{
   return ComplexDouble< T >::mul( a, b );
}

template< typename T >
__cuda_callable__
constexpr __xdw_inline__ bool
operator==( const ComplexDouble< T >& a, const ComplexDouble< T >& b )
{
   return a.re_h() == b.re_h() && a.re_l() == b.re_l() && a.im_h() == b.im_h() && a.im_l() == b.im_l();
}

template< typename T >
__cuda_callable__
constexpr __xdw_inline__ ComplexDouble< T >
conj( const ComplexDouble< T >& z )
{
   return ComplexDouble< T >( z.re_h(), z.re_l(), -z.im_h(), -z.im_l() );
}

template< typename T >
__cuda_callable__
constexpr __xdw_inline__ ComplexDouble< T >
real( const ComplexDouble< T >& z )
{
   return ComplexDouble< T >( z.re_h(), z.re_l(), T( 0 ), T( 0 ) );
}

template< typename T >
__cuda_callable__
constexpr __xdw_inline__ ComplexDouble< T >
imag( const ComplexDouble< T >& z )
{
   return ComplexDouble< T >( z.im_h(), z.im_l(), T( 0 ), T( 0 ) );
}

// |z|^2, real-valued, no sqrt.
template< typename T, XDW_ARTH::AddMode Add = XDW_ARTH::kAddMode >
__cuda_callable__
constexpr __xdw_inline__ ComplexDouble< T >
norm( const ComplexDouble< T >& z )
{
   T rh, rl;
   XDW_ARTH::DWMulAdd< T, Add >( z.re_h(), z.re_l(), z.re_h(), z.re_l(),
                                  z.im_h(), z.im_l(), z.im_h(), z.im_l(), &rh, &rl );
   return ComplexDouble< T >( rh, rl, T( 0 ), T( 0 ) );
}

#endif
