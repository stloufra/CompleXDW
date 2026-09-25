#ifndef XDWTRAITS_H
#define XDWTRAITS_H

#pragma once

#include <cfloat>
#include <concepts>
#include <type_traits>
#include <utility>

#if defined( __CUDACC__ ) || defined( __HIPCC__ )
#define XDW_CUDA_CALLABLE \
__device__             \
__host__
#else
#define XDW_CUDA_CALLABLE
#endif

#if !defined( NDEBUG ) // not in debug mode
#define XDW_INLINE inline
#elif defined( __CUDACC__ ) || defined( __HIPCC__ )
#define XDW_INLINE __forceinline__
#elif defined( _MSC_VER )
#define XDW_INLINE __forceinline
#elif defined( __GNUC__ ) || defined( __clang__ )
#define XDW_INLINE __attribute__( ( always_inline ) ) inline
#else
#define XDW_INLINE inline
#endif

#if !defined( __CUDA_ARCH__ ) && !defined( __HIP_DEVICE_COMPILE__ )

namespace XDW_ARTH::detail {

#if defined( __FAST_MATH__ )
inline constexpr bool fast_math_enabled = true;
#else
inline constexpr bool fast_math_enabled = false;
#endif

}

static_assert( !XDW_ARTH::detail::fast_math_enabled,
               "CompleXDW's round-to-nearest error-free transforms are unsound under "
               "-ffast-math/-Ofast; build without fast-math." );

#if defined( __FLT_EVAL_METHOD__ )
static_assert( __FLT_EVAL_METHOD__ != 2,
               "FLT_EVAL_METHOD == 2 means excess-precision (x87) intermediates, which breaks "
               "the round-to-nearest error-free transforms; build with SSE2 math instead." );
#endif

// Without a hardware-FMA target, std::fma (fma_rn) is a libm call, emulated in software on CPUs without FMA.
#if !defined( __FMA__ ) && !defined( __ARM_FEATURE_FMA ) && !defined( __FP_FAST_FMA ) \
    && !( defined( _MSC_VER ) && defined( __AVX2__ ) ) && !defined( XDW_ALLOW_SOFTWARE_FMA )
#if defined( _MSC_VER ) && !defined( __clang__ )
#pragma message( "CompleXDW: target has no hardware FMA, so every fma_rn is a slow libm call. Build with /arch:AVX2; define XDW_ALLOW_SOFTWARE_FMA to silence." )
#else
#warning "CompleXDW: target has no hardware FMA, so every fma_rn is a slow libm call. Build with -march=native (or -mfma on x86-64); define XDW_ALLOW_SOFTWARE_FMA to silence."
#endif
#endif

#endif

namespace XDW_ARTH {

enum class AddMode { Accurate, Madd, Sloppy };
enum class NormMode { Normalized, Unnormalized };
enum class DivMode { Div2, Div3 };

// A compiler SIMD vector of float/double (clang ext_vector_type, GCC vector_size): each lane is an
// independent number. Host only.
template< typename T >
concept XDWVector = !std::is_arithmetic_v< T > && requires( T v ) { v[ 0 ]; }
                    && std::floating_point< std::remove_cvref_t< decltype( std::declval< T >()[ 0 ] ) > >;

template< typename T >
concept XDWReal = std::floating_point< T > || XDWVector< T >;

namespace detail {
template< typename T >
struct lane { using type = T; };

template< XDWVector T >
struct lane< T > { using type = std::remove_cvref_t< decltype( std::declval< T >()[ 0 ] ) >; };
}

template< XDWReal T >
using lane_t = typename detail::lane< T >::type;

template< XDWReal T >
inline constexpr int lanes = sizeof( T ) / sizeof( lane_t< T > );

// c in every lane.
template< XDWReal T >
XDW_CUDA_CALLABLE constexpr T splat( lane_t< T > c )
{
   if constexpr( XDWVector< T > )
      return T{} + c;
   else
      return c;
}

// && and || for scalar comparisons (bool), & and | lane by lane for vector masks.
template< typename M >
XDW_CUDA_CALLABLE constexpr M mask_and( M a, M b )
{
   if constexpr( std::is_same_v< M, bool > )
      return a && b;
   else
      return a & b;
}

template< typename M >
XDW_CUDA_CALLABLE constexpr M mask_or( M a, M b )
{
   if constexpr( std::is_same_v< M, bool > )
      return a || b;
   else
      return a | b;
}

}

// Compile time set as -DXDW_ADD_MODE=<var> -DXDW_NORM_MODE=<var> -DXDW_DIV_MODE=<var>
// XDW_ADD_MODE:  A = Accurate, M = Madd, S = Sloppy
// XDW_NORM_MODE: N = Normalized, U = Unnormalized
// XDW_DIV_MODE:  2 = Div2, 3 = Div3
// Unset macros default to Madd, Normalized, Div3
// XDW_DEKKER_TWO_PROD: TwoProd via Dekker's split instead of one FMA (host only)

#if !defined( XDW_ADD_MODE )
#define XDW_ADD_MODE M
#endif

#if !defined( XDW_NORM_MODE )
#define XDW_NORM_MODE N
#endif

#if !defined( XDW_DIV_MODE )
#define XDW_DIV_MODE 3
#endif

#define XDW_STRINGIFY_( x ) #x
#define XDW_STRINGIFY( x ) XDW_STRINGIFY_( x )

namespace XDW_ARTH::detail {

constexpr bool is_mode( const char* s, char allowed )
{
   return s[ 0 ] == allowed && s[ 1 ] == '\0';
}

}

static_assert( XDW_ARTH::detail::is_mode( XDW_STRINGIFY( XDW_ADD_MODE ), 'A' )
            || XDW_ARTH::detail::is_mode( XDW_STRINGIFY( XDW_ADD_MODE ), 'M' )
            || XDW_ARTH::detail::is_mode( XDW_STRINGIFY( XDW_ADD_MODE ), 'S' ),
            "XDW_ADD_MODE must be A (Accurate), M (Madd), or S (Sloppy)" );

static_assert( XDW_ARTH::detail::is_mode( XDW_STRINGIFY( XDW_NORM_MODE ), 'N' )
            || XDW_ARTH::detail::is_mode( XDW_STRINGIFY( XDW_NORM_MODE ), 'U' ),
            "XDW_NORM_MODE must be N (Normalized) or U (Unnormalized)" );

static_assert( XDW_ARTH::detail::is_mode( XDW_STRINGIFY( XDW_DIV_MODE ), '2' )
            || XDW_ARTH::detail::is_mode( XDW_STRINGIFY( XDW_DIV_MODE ), '3' ),
            "XDW_DIV_MODE must be 2 (Div2) or 3 (Div3)" );

#undef XDW_STRINGIFY
#undef XDW_STRINGIFY_

#define XDW_ADD_TAG_A ::XDW_ARTH::AddMode::Accurate
#define XDW_ADD_TAG_M ::XDW_ARTH::AddMode::Madd
#define XDW_ADD_TAG_S ::XDW_ARTH::AddMode::Sloppy

#define XDW_NORM_TAG_N ::XDW_ARTH::NormMode::Normalized
#define XDW_NORM_TAG_U ::XDW_ARTH::NormMode::Unnormalized

#define XDW_DIV_TAG_2 ::XDW_ARTH::DivMode::Div2
#define XDW_DIV_TAG_3 ::XDW_ARTH::DivMode::Div3

#define XDW_TOKEN_PASTE_( a, b ) a##b
#define XDW_TOKEN_PASTE( a, b ) XDW_TOKEN_PASTE_( a, b )

namespace XDW_ARTH {

inline constexpr AddMode kAddMode = XDW_TOKEN_PASTE( XDW_ADD_TAG_, XDW_ADD_MODE );
inline constexpr NormMode kNormMode = XDW_TOKEN_PASTE( XDW_NORM_TAG_, XDW_NORM_MODE );
inline constexpr DivMode kDivMode = XDW_TOKEN_PASTE( XDW_DIV_TAG_, XDW_DIV_MODE );

}

#undef XDW_TOKEN_PASTE
#undef XDW_TOKEN_PASTE_

#endif  //XDWTRAITS_H
