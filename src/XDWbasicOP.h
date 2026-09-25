#ifndef XDW_BASIC_OPS_H
#define XDW_BASIC_OPS_H

#pragma once

#include <concepts>
#include <limits>
#include <type_traits>
#include <cmath>

#include "XDWTraits.h"

#if defined( __GNUC__ ) && defined( __has_builtin )
#if __has_builtin( __builtin_assoc_barrier )
#define XDW_HAS_ASSOC_BARRIER
#endif
#endif

#if defined( __HIPCC__ )
#include <hip/hip_runtime.h>
#elif defined( __CUDACC__ )
#include <cuda_runtime.h>
#endif


namespace XDW_ARTH{

//implementation of basic operations to ensure round to nearest in CUDA/HIP, and no FMA contraction on the host

template< std::floating_point T >
XDW_CUDA_CALLABLE
static constexpr XDW_INLINE T
add_rn( const T x, const T y )
{
#if defined( __HIP_DEVICE_COMPILE__ ) || defined( __CUDA_ARCH__ )
   if constexpr( std::is_same_v< T, double > ) {
      return __dadd_rn( x, y );
   }
   else if constexpr( std::is_same_v< T, float > ) {
      return __fadd_rn( x, y );
   }
#else
#if defined( __clang__ )
#pragma clang fp contract( off )
#endif
   return x + y;
#endif
}

template< std::floating_point T >
XDW_CUDA_CALLABLE
static constexpr XDW_INLINE T
mul_rn( const T x, const T y )
{
#if defined( __HIP_DEVICE_COMPILE__ ) || defined( __CUDA_ARCH__ )
   if constexpr( std::is_same_v< T, double > ) {
      return __dmul_rn( x, y );
   }
   else if constexpr( std::is_same_v< T, float > ) {
      return __fmul_rn( x, y );
   }
#else
#if defined( __clang__ )
#pragma clang fp contract( off )
   return x * y;
#elif defined( XDW_HAS_ASSOC_BARRIER )
   // g++ has no per-region fp-contract; the barrier keeps the product rounded so it can't be fused into an add.
   return __builtin_assoc_barrier( x * y );
#else
   return x * y;
#endif
#endif
}

template< std::floating_point T >
XDW_CUDA_CALLABLE
static constexpr XDW_INLINE T
div_rn( const T x, const T y )
{
#if defined( __HIP_DEVICE_COMPILE__ ) || defined( __CUDA_ARCH__ )
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
#if defined( __HIP_DEVICE_COMPILE__ ) || defined( __CUDA_ARCH__ )
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
