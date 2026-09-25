#ifndef XDW_SPAN_H
#define XDW_SPAN_H

#include <concepts>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "XDW.h"

// Non-owning structure-of-arrays view 
// (re_h[i] + re_l[i]) + i*(im_h[i]+ im_l[i]).
// For vectorization and GPU (accepts device pointers)
template< typename T >
requires std::floating_point< std::remove_const_t< T > >
struct XDWSpan
{
  using value_type = XDW< std::remove_const_t< T > >;

  T* re_h = nullptr;
  T* re_l = nullptr;
  T* im_h = nullptr;
  T* im_l = nullptr;
  std::size_t size = 0;

  XDW_CUDA_CALLABLE
  constexpr XDWSpan() = default;

  XDW_CUDA_CALLABLE
  constexpr XDWSpan( T* re_h_, T* re_l_, T* im_h_, T* im_l_, std::size_t size_ )
  : re_h( re_h_ ), re_l( re_l_ ), im_h( im_h_ ), im_l( im_l_ ), size( size_ ) {}

  // Mutable view -> read-only view.
  template< typename U >
  requires std::same_as< const U, T >
  XDW_CUDA_CALLABLE
  constexpr XDWSpan( const XDWSpan< U >& other )
  : re_h( other.re_h ), re_l( other.re_l ), im_h( other.im_h ), im_l( other.im_l ), size( other.size ) {}

  XDW_CUDA_CALLABLE
  constexpr value_type load( std::size_t i ) const { return value_type( re_h[ i ], re_l[ i ], im_h[ i ], im_l[ i ] ); }

  XDW_CUDA_CALLABLE
  constexpr void store( std::size_t i, const value_type& z ) const
  requires( !std::is_const_v< T > )
  {
    re_h[ i ] = z.re_h();
    re_l[ i ] = z.re_l();
    im_h[ i ] = z.im_h();
    im_l[ i ] = z.im_l();
  }
};

namespace XDW_ARTH::detail {

template< typename T >
void check_same_size( const char* op, const XDWSpan< T >& out,
                      const XDWSpan< const T >& a, const XDWSpan< const T >& b )
{
  if( a.size == out.size && b.size == out.size )
    return;
  throw std::invalid_argument( std::string( op ) + ": span sizes differ (out " + std::to_string( out.size )
                               + ", a " + std::to_string( a.size ) + ", b " + std::to_string( b.size ) + ")" );
}

}

// out[i] = a[i] * b[i]. out may be a or b (in place).
template< XDW_ARTH::AddMode Add = XDW_ARTH::kAddMode, XDW_ARTH::NormMode Norm = XDW_ARTH::kNormMode,
          std::floating_point T >
void mul( XDWSpan< T > out, std::type_identity_t< XDWSpan< const T > > a,
          std::type_identity_t< XDWSpan< const T > > b )
{
  XDW_ARTH::detail::check_same_size( "mul", out, a, b );
  for( std::size_t i = 0; i < out.size; ++i )
    XDW_ARTH::XDWmul< T, Add, Norm >( a.re_h[ i ], a.re_l[ i ], a.im_h[ i ], a.im_l[ i ],
                                       b.re_h[ i ], b.re_l[ i ], b.im_h[ i ], b.im_l[ i ],
                                       &out.re_h[ i ], &out.re_l[ i ], &out.im_h[ i ], &out.im_l[ i ] );
}

// out[i] = a[i] / b[i]. out may be a or b (in place).
template< XDW_ARTH::DivMode Div = XDW_ARTH::kDivMode, XDW_ARTH::AddMode Add = XDW_ARTH::kAddMode,
          XDW_ARTH::NormMode Norm = XDW_ARTH::kNormMode, std::floating_point T >
void div( XDWSpan< T > out, std::type_identity_t< XDWSpan< const T > > a,
          std::type_identity_t< XDWSpan< const T > > b )
{
  XDW_ARTH::detail::check_same_size( "div", out, a, b );
  for( std::size_t i = 0; i < out.size; ++i )
    XDW_ARTH::XDWDiv< T, Div, Add, Norm >( a.re_h[ i ], a.re_l[ i ], a.im_h[ i ], a.im_l[ i ],
                                            b.re_h[ i ], b.re_l[ i ], b.im_h[ i ], b.im_l[ i ],
                                            &out.re_h[ i ], &out.re_l[ i ], &out.im_h[ i ], &out.im_l[ i ] );
}

#endif
