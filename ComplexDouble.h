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

  XDW_CUDA_CALLABLE
  constexpr ComplexDouble() = default;

  XDW_CUDA_CALLABLE
  constexpr ComplexDouble( const ComplexDouble& other ) = default;

  XDW_CUDA_CALLABLE
  constexpr ComplexDouble( ComplexDouble&& other ) noexcept = default;

  XDW_CUDA_CALLABLE
  constexpr ComplexDouble& operator=( const ComplexDouble& other ) = default;

  XDW_CUDA_CALLABLE
  constexpr ComplexDouble& operator=( ComplexDouble&& other ) noexcept = default;

  XDW_CUDA_CALLABLE
  constexpr ComplexDouble( const std::complex< T >& c );

  XDW_CUDA_CALLABLE
  constexpr ComplexDouble( T re );

  XDW_CUDA_CALLABLE
  constexpr ComplexDouble( T re, T im );

  XDW_CUDA_CALLABLE
  constexpr ComplexDouble( T re_h, T re_l, T im_h, T im_l );

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
  constexpr ComplexDouble< T >
  operator+() const;

  XDW_CUDA_CALLABLE
  constexpr ComplexDouble< T >
  operator-() const;

  XDW_CUDA_CALLABLE
  constexpr ComplexDouble< T >&
  operator+=( const ComplexDouble< T >& other );

  XDW_CUDA_CALLABLE
  constexpr ComplexDouble< T >&
  operator-=( const ComplexDouble< T >& other );

  XDW_CUDA_CALLABLE
  constexpr ComplexDouble< T >&
  operator*=( const ComplexDouble< T >& other );

  template< XDW_ARTH::AddMode Add = XDW_ARTH::kAddMode >
  XDW_CUDA_CALLABLE
  constexpr static ComplexDouble< T >
  add( const ComplexDouble< T >& a, const ComplexDouble< T >& b );

  template< XDW_ARTH::AddMode Add = XDW_ARTH::kAddMode >
  XDW_CUDA_CALLABLE
  constexpr static ComplexDouble< T >
  sub( const ComplexDouble< T >& a, const ComplexDouble< T >& b );

  template< XDW_ARTH::AddMode Add = XDW_ARTH::kAddMode, XDW_ARTH::NormMode Norm = XDW_ARTH::kNormMode >
  XDW_CUDA_CALLABLE
  constexpr static ComplexDouble< T >
  mul( const ComplexDouble< T >& a, const ComplexDouble< T >& b );
};

template< typename T >
XDW_CUDA_CALLABLE
constexpr ComplexDouble< T >::ComplexDouble( T re_h_, T re_l_, T im_h_, T im_l_ )
   : data{ re_h_, re_l_, im_h_, im_l_ }
{}

template< typename T >
XDW_CUDA_CALLABLE
constexpr ComplexDouble< T >::ComplexDouble( const std::complex< T >& c )
{
   data[ 0 ] = c.real();
   data[ 1 ] = static_cast< T >(0.0F);
   data[ 2 ] = c.imag();
   data[ 3 ] = static_cast< T >(0.0F);
}

template< typename T >
XDW_CUDA_CALLABLE
constexpr ComplexDouble< T >::ComplexDouble( T re_ )
   : data{ re_, T(0), T(0), T(0) }
{}

template< typename T >
XDW_CUDA_CALLABLE
constexpr ComplexDouble< T >::ComplexDouble( T re_, T im_)
   : data{ re_, T(0), im_, T(0) }
{}

template< typename T >
XDW_CUDA_CALLABLE
constexpr ComplexDouble< T >
ComplexDouble< T >::operator+() const
{
   return *this;
}

template< typename T >
XDW_CUDA_CALLABLE
constexpr ComplexDouble< T >
ComplexDouble< T >::operator-() const
{
   return ComplexDouble< T >( -data[ 0 ], -data[ 1 ], -data[ 2 ], -data[ 3 ] );
}

template< typename T >
template< XDW_ARTH::AddMode Add >
XDW_CUDA_CALLABLE
constexpr XDW_INLINE ComplexDouble< T >
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
XDW_CUDA_CALLABLE
constexpr XDW_INLINE ComplexDouble< T >
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
XDW_CUDA_CALLABLE
constexpr XDW_INLINE ComplexDouble< T >
ComplexDouble< T >::mul( const ComplexDouble< T >& a, const ComplexDouble< T >& b )
{
   T reh, rel, imh, iml;
   XDW_ARTH::XDWmul< T, Add, Norm >( a.re_h(), a.re_l(), a.im_h(), a.im_l(),
                                      b.re_h(), b.re_l(), b.im_h(), b.im_l(), &reh, &rel, &imh, &iml );
   return ComplexDouble< T >( reh, rel, imh, iml );
}



template< typename T >
XDW_CUDA_CALLABLE
constexpr XDW_INLINE ComplexDouble< T >&
ComplexDouble< T >::operator+=( const ComplexDouble< T >& other )
{
   return *this = *this + other;
}

template< typename T >
XDW_CUDA_CALLABLE
constexpr XDW_INLINE ComplexDouble< T >&
ComplexDouble< T >::operator-=( const ComplexDouble< T >& other )
{
   return *this = *this - other;
}

template< typename T >
XDW_CUDA_CALLABLE
constexpr XDW_INLINE ComplexDouble< T >&
ComplexDouble< T >::operator*=( const ComplexDouble< T >& other )
{
   return *this = *this * other;
}

template< typename T >
XDW_CUDA_CALLABLE
constexpr XDW_INLINE ComplexDouble< T >
operator+( const ComplexDouble< T >& a, const ComplexDouble< T >& b )
{
   return ComplexDouble< T >::add( a, b );
}

template< typename T >
XDW_CUDA_CALLABLE
constexpr XDW_INLINE ComplexDouble< T >
operator-( const ComplexDouble< T >& a, const ComplexDouble< T >& b )
{
   return ComplexDouble< T >::sub( a, b );
}

template< typename T >
XDW_CUDA_CALLABLE
constexpr XDW_INLINE ComplexDouble< T >
operator*( const ComplexDouble< T >& a, const ComplexDouble< T >& b )
{
   return ComplexDouble< T >::mul( a, b );
}

template< typename T >
XDW_CUDA_CALLABLE
constexpr XDW_INLINE bool
operator==( const ComplexDouble< T >& a, const ComplexDouble< T >& b )
{
   return a.re_h() == b.re_h() && a.re_l() == b.re_l() && a.im_h() == b.im_h() && a.im_l() == b.im_l();
}

template< typename T >
XDW_CUDA_CALLABLE
constexpr XDW_INLINE ComplexDouble< T >
conj( const ComplexDouble< T >& z )
{
   return ComplexDouble< T >( z.re_h(), z.re_l(), -z.im_h(), -z.im_l() );
}

template< typename T >
XDW_CUDA_CALLABLE
constexpr XDW_INLINE ComplexDouble< T >
real( const ComplexDouble< T >& z )
{
   return ComplexDouble< T >( z.re_h(), z.re_l(), T( 0 ), T( 0 ) );
}

template< typename T >
XDW_CUDA_CALLABLE
constexpr XDW_INLINE ComplexDouble< T >
imag( const ComplexDouble< T >& z )
{
   return ComplexDouble< T >( z.im_h(), z.im_l(), T( 0 ), T( 0 ) );
}

// |z|^2 = z * z', real-valued, no sqrt.
template< typename T, XDW_ARTH::AddMode Add = XDW_ARTH::kAddMode, XDW_ARTH::NormMode Norm = XDW_ARTH::kNormMode >
XDW_CUDA_CALLABLE
constexpr XDW_INLINE ComplexDouble< T >
norm( const ComplexDouble< T >& z )
{
   T rh, rl;
   XDW_ARTH::DWPowAdd< T, Add, Norm >( z.re_h(), z.re_l(), z.im_h(), z.im_l(), &rh, &rl );
   return ComplexDouble< T >( rh, rl, T( 0 ), T( 0 ) );
}

#endif
