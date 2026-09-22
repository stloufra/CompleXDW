#ifndef XDW_BASIC_OPS_H
#define XDW_BASIC_OPS_H

#pragma once

#include <concepts>
#include <limits>
#include <type_traits>
#include <cmath>

#include "XDWTraits.h"

#if defined( __CUDACC__ )
#include <cuda_runtime.h>
#endif


namespace XDW_ARTH{

//implementation of basic operations to ensure round to nearest in CUDA

template< std::floating_point T >
XDW_CUDA_CALLABLE
static constexpr XDW_INLINE T
add_rn( const T x, const T y )
{
#if defined __CUDA_ARCH__
   if constexpr( std::is_same_v< T, double > ) {
      return __dadd_rn( x, y );
   }
   else if constexpr( std::is_same_v< T, float > ) {
      return __fadd_rn( x, y );
   }
#else
   return x + y;
#endif
}

template< std::floating_point T >
XDW_CUDA_CALLABLE
static constexpr XDW_INLINE T
mul_rn( const T x, const T y )
{
#if defined __CUDA_ARCH__
   if constexpr( std::is_same_v< T, double > ) {
      return __dmul_rn( x, y );
   }
   else if constexpr( std::is_same_v< T, float > ) {
      return __fmul_rn( x, y );
   }
#else
   return x * y;
#endif
}

template< std::floating_point T >
XDW_CUDA_CALLABLE
static constexpr XDW_INLINE T
div_rn( const T x, const T y )
{
#if defined __CUDA_ARCH__
   if constexpr( std::is_same_v< T, double > ) {
      return __ddiv_rn( x, y );
   }
   else if constexpr( std::is_same_v< T, float > ) {
      return __fdiv_rn( x, y );
   }
#else
   return x / y;
#endif
}

template< std::floating_point T >
XDW_CUDA_CALLABLE
static constexpr XDW_INLINE T
fma_rn( const T x, const T y, const T z )
{
#if defined __CUDA_ARCH__
   if constexpr( std::is_same_v< T, double > ) {
      return __fma_rn( x, y, z );
   }
   else if constexpr( std::is_same_v< T, float > ) {
      return __fmaf_rn( x, y, z );
   }
#else
    return std::fma( x, y, z );
#endif
}

}

#endif  //XDW_BASIC_OPS_H
