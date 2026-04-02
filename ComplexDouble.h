#ifndef COMPLEX_DOUBLE_H
#define COMPLEX_DOUBLE_H

#include "src/XDWerrorFree.h"
#include "src/XDWArith.h"
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
  constexpr static ComplexDouble< T >
  add( const ComplexDouble< T >& a, const ComplexDouble< T >& b );

  __cuda_callable__
  constexpr static ComplexDouble< T >
  sub( const ComplexDouble< T >& a, const ComplexDouble< T >& b );

  __cuda_callable__
  constexpr static ComplexDouble< T >
  mulnorm( const ComplexDouble< T >& a, const ComplexDouble< T >& b );

  __cuda_callable__
  constexpr static ComplexDouble< T >
  mulSloppyUnnorm( const ComplexDouble< T >& a, const ComplexDouble< T >& b );

  __cuda_callable__
  constexpr static ComplexDouble< T >
  mulAccurateUnnorm( const ComplexDouble< T >& a, const ComplexDouble< T >& b );
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
__cuda_callable__
constexpr ComplexDouble< T >
ComplexDouble< T >::add( const ComplexDouble< T >& a, const ComplexDouble< T >& b )
{
   T reh, rel, imh, iml;
   XDW_ARTH::maddDWPlusDW( a.data[ 0 ], a.data[ 1 ], b.data[ 0 ], b.data[ 1 ], &reh, &rel );
   XDW_ARTH::maddDWPlusDW( a.data[ 2 ], a.data[ 3 ], b.data[ 2 ], b.data[ 3 ], &imh, &iml );
   return ComplexDouble< T >( reh, rel, imh, iml );
}

template< typename T >
__cuda_callable__
constexpr ComplexDouble< T >
ComplexDouble< T >::sub( const ComplexDouble< T >& a, const ComplexDouble< T >& b )
{
   T reh, rel, imh, iml;
   XDW_ARTH::maddDWPlusDW( a.data[ 0 ], a.data[ 1 ], -b.data[ 0 ], -b.data[ 1 ], &reh, &rel );
   XDW_ARTH::maddDWPlusDW( a.data[ 2 ], a.data[ 3 ], -b.data[ 2 ], -b.data[ 3 ], &imh, &iml );
   return ComplexDouble< T >( reh, rel, imh, iml );
}

template< typename T >
__cuda_callable__
constexpr ComplexDouble< T >
ComplexDouble< T >::mulnorm( const ComplexDouble< T >& a, const ComplexDouble< T >& b )
{
   T reh, rel, imh, iml;
   XDW_ARTH::ComplexDWMulNorm( a.data[ 0 ], a.data[ 1 ], a.data[ 2 ], a.data[ 3 ],
                     b.data[ 0 ], b.data[ 1 ], b.data[ 2 ], b.data[ 3 ],
                     &reh, &rel, &imh, &iml );
   return ComplexDouble< T >( reh, rel, imh, iml );
}

template< typename T >
__cuda_callable__
constexpr ComplexDouble< T >
ComplexDouble< T >::mulSloppyUnnorm( const ComplexDouble< T >& a, const ComplexDouble< T >& b )
{
   T reh, rel, imh, iml;
   XDW_ARTH::ComplexDWMulSloppyUnnorm( a.data[ 0 ], a.data[ 1 ], a.data[ 2 ], a.data[ 3 ],
                     b.data[ 0 ], b.data[ 1 ], b.data[ 2 ], b.data[ 3 ],
                     &reh, &rel, &imh, &iml );
   return ComplexDouble< T >( reh, rel, imh, iml );
}


template< typename T >
__cuda_callable__
constexpr ComplexDouble< T >
ComplexDouble< T >::mulAccurateUnnorm( const ComplexDouble< T >& a, const ComplexDouble< T >& b )
{
   T reh, rel, imh, iml;
   XDW_ARTH::ComplexDWMulAccurateUnnorm( a.data[ 0 ], a.data[ 1 ], a.data[ 2 ], a.data[ 3 ],
                     b.data[ 0 ], b.data[ 1 ], b.data[ 2 ], b.data[ 3 ],
                     &reh, &rel, &imh, &iml );
   return ComplexDouble< T >( reh, rel, imh, iml );
}



template< typename T >
__cuda_callable__
constexpr ComplexDouble< T >
operator+( const ComplexDouble< T >& a, const ComplexDouble< T >& b )
{
   return ComplexDouble< T >::add( a, b );
}

template< typename T >
__cuda_callable__
constexpr ComplexDouble< T >
operator-( const ComplexDouble< T >& a, const ComplexDouble< T >& b )
{
   return ComplexDouble< T >::sub( a, b );
}

//define muliplication

template< typename T >
__cuda_callable__
constexpr ComplexDouble< T >
operator*( const ComplexDouble< T >& a, const ComplexDouble< T >& b )
{
   return ComplexDouble< T >::mulnorm( a, b );
}

template< typename T >
__cuda_callable__
constexpr ComplexDouble< T >
mul_sloppy_unnnorm( const ComplexDouble< T >& a, const ComplexDouble< T >& b )
{
   return ComplexDouble< T >::mulSloppyUnnorm( a, b );
}

template< typename T >
__cuda_callable__
constexpr ComplexDouble< T >
mul_accurate_unnnorm( const ComplexDouble< T >& a, const ComplexDouble< T >& b )
{
   return ComplexDouble< T >::mulAccurateUnnorm( a, b );
}


#endif
