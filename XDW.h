#ifndef XDW_H
#define XDW_H

#include "src/XDWerrorFree.h"
#include "src/XDWarith.h"
#include "src/XDWTraits.h"
#include "DW.h"
#include <complex>
#include <type_traits>

template< typename T >
class alignas( 4 * sizeof( T ) ) XDW
{

  static_assert( std::is_same_v< T, float > || std::is_same_v< T, double >
                 ,"XDW<T> can only be instantiated with float or double." );

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

  XDW_CUDA_CALLABLE
  constexpr XDW( const std::complex< T >& c );

  // Plain numbers go through DW, so a double into XDW<float> keeps its low part.
  template< typename U >
  requires std::is_arithmetic_v< U >
  XDW_CUDA_CALLABLE
  constexpr XDW( U re ) : XDW( DW< T >( re ), DW< T >( T( 0 ), T( 0 ) ) ) {}

  template< typename U, typename V >
  requires std::is_arithmetic_v< U > && std::is_arithmetic_v< V >
  XDW_CUDA_CALLABLE
  constexpr XDW( U re, V im ) : XDW( DW< T >( re ), DW< T >( im ) ) {}

  XDW_CUDA_CALLABLE
  constexpr XDW( const DW< T >& re, const DW< T >& im = DW< T >( T( 0 ), T( 0 ) ) )
  : data{ re.hi(), re.lo(), im.hi(), im.lo() } {}

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
constexpr XDW< T >::XDW( const std::complex< T >& c )
{
   data[ 0 ] = c.real();
   data[ 1 ] = static_cast< T >(0.0F);
   data[ 2 ] = c.imag();
   data[ 3 ] = static_cast< T >(0.0F);
}

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

template< typename T >
XDW_CUDA_CALLABLE
constexpr XDW_INLINE bool
operator==( const XDW< T >& a, const XDW< T >& b )
{
   return a.re_h() == b.re_h() && a.re_l() == b.re_l() && a.im_h() == b.im_h() && a.im_l() == b.im_l();
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

#endif
