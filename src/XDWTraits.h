#ifndef XDWTRAITS_H
#define XDWTRAITS_H

#pragma once

#if defined( __CUDACC__ )
#define __cuda_callable__ \
__device__             \
__host__
#else
#define __cuda_callable__
#endif

#if defined( __CUDACC__ )
#define __xdw_inline__ __forceinline__
#elif defined( _MSC_VER )
#define __xdw_inline__ __forceinline
#elif defined( __GNUC__ ) || defined( __clang__ )
#define __xdw_inline__ __attribute__( ( always_inline ) ) inline
#else
#define __xdw_inline__ inline
#endif

namespace XDW_ARTH {

enum class AddMode { Accurate, Madd, Sloppy };
enum class NormMode { Normalized, Unnormalized };

}

// Compile time set as -DXDW_ADD_MODE=<var> -DXDW_NORM_MODE=<var>
// XDW_ADD_MODE:  A = Accurate, M = Madd, S = Sloppy
// XDW_NORM_MODE: N = Normalized, U = Unnormalized
// Unset macros default to Madd, Normalized

#if !defined( XDW_ADD_MODE )
#define XDW_ADD_MODE M
#endif

#if !defined( XDW_NORM_MODE )
#define XDW_NORM_MODE N
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

#undef XDW_STRINGIFY
#undef XDW_STRINGIFY_

#define XDW_ADD_TAG_A ::XDW_ARTH::AddMode::Accurate
#define XDW_ADD_TAG_M ::XDW_ARTH::AddMode::Madd
#define XDW_ADD_TAG_S ::XDW_ARTH::AddMode::Sloppy

#define XDW_NORM_TAG_N ::XDW_ARTH::NormMode::Normalized
#define XDW_NORM_TAG_U ::XDW_ARTH::NormMode::Unnormalized

#define XDW_TOKEN_PASTE_( a, b ) a##b
#define XDW_TOKEN_PASTE( a, b ) XDW_TOKEN_PASTE_( a, b )

namespace XDW_ARTH {

inline constexpr AddMode kAddMode = XDW_TOKEN_PASTE( XDW_ADD_TAG_, XDW_ADD_MODE );
inline constexpr NormMode kNormMode = XDW_TOKEN_PASTE( XDW_NORM_TAG_, XDW_NORM_MODE );

}

#undef XDW_TOKEN_PASTE
#undef XDW_TOKEN_PASTE_

#endif  //XDWTRAITS_H
