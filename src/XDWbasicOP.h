#ifndef XDW_BASIC_OPS_H
#define XDW_BASIC_OPS_H

#pragma once

#include <limits>
#include <cmath>

#if defined( __CUDACC__ )
#include <cuda_runtime.h>
#endif


namespace XDW_ARTH{

//implementation of basic operations to ensure round to nearest in CUDA

#if defined( __CUDACC__ )
#define __xdw_inline__ __forceinline__
#elif defined( _MSC_VER )
#define __xdw_inline__ __forceinline
#elif defined( __GNUC__ ) || defined( __clang__ )
#define __xdw_inline__ __attribute__( ( always_inline ) ) inline
#else
#define __xdw_inline__ inline
#endif

#if defined( __CUDACC__ )
#define __cuda_callable__ \
__device__             \
__host__
#else
#define __cuda_callable__
#endif

#define FLOAT_TEMPLATE_GUARD \
template< typename T, std::enable_if_t< std::is_floating_point_v< T >, int > = 0 >


FLOAT_TEMPLATE_GUARD
__cuda_callable__
static constexpr __xdw_inline__ T
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

FLOAT_TEMPLATE_GUARD
__cuda_callable__
static constexpr __xdw_inline__ T
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

FLOAT_TEMPLATE_GUARD
__cuda_callable__
static constexpr __xdw_inline__ T
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

FLOAT_TEMPLATE_GUARD
__cuda_callable__
static constexpr __xdw_inline__ T
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
