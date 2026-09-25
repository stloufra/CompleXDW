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

template< XDWReal T >
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
   else {
      static_assert( std::is_same_v< T, float >, "XDW: SIMD vector T is host-only" );
   }
#else
#if defined( __clang__ )
#pragma clang fp contract( off )
#endif
   return x + y;
#endif
}

template< XDWReal T >
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
   else {
      static_assert( std::is_same_v< T, float >, "XDW: SIMD vector T is host-only" );
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

template< XDWReal T >
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
   else {
      static_assert( std::is_same_v< T, float >, "XDW: SIMD vector T is host-only" );
   }
#else
   return x / y;
#endif
}

template< XDWReal T >
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
   else {
      static_assert( std::is_same_v< T, float >, "XDW: SIMD vector T is host-only" );
   }
#else
   if constexpr( XDWVector< T > ) {
#if defined( __clang__ )
      return __builtin_elementwise_fma( x, y, z );
#else
      // g++ turns the lane loop into vector FMAs.
      T r;
      for( int i = 0; i < lanes< T >; ++i ) r[ i ] = std::fma( x[ i ], y[ i ], z[ i ] );
      return r;
#endif
   }
   else
      return std::fma( x, y, z );
#endif
}

}

#endif  //XDW_BASIC_OPS_H
