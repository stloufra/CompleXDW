#ifndef MPFR_SUPPORT_H
#define MPFR_SUPPORT_H

// Exact references for the unit tests: random double-word inputs and MPFR arithmetic on them.

#include <cmath>
#include <limits>
#include <random>
#include <type_traits>

#include <gmp.h>
#include <mpfr.h>

#include "XDW.h"

constexpr int MPFR_BITS = 1024;
constexpr int SAMPLES = 200000;

// u^2 of T, u = 2^-p.
template< typename T >
constexpr double U2 = 1.0 / ( double( 1ull << std::numeric_limits< T >::digits ) * double( 1ull << std::numeric_limits< T >::digits ) );

struct Big
{
   mpfr_t v;
   Big() { mpfr_init2( v, MPFR_BITS ); }
   ~Big() { mpfr_clear( v ); }
   Big( const Big& ) = delete;
   Big& operator=( const Big& ) = delete;
   operator mpfr_ptr() { return v; }
   operator mpfr_srcptr() const { return v; }
};

template< typename T >
T get( mpfr_srcptr x )
{
   if constexpr( std::is_same_v< T, float > )
      return mpfr_get_flt( x, MPFR_RNDN );
   else
      return mpfr_get_d( x, MPFR_RNDN );
}

// Round to hi, then round the remainder to lo: a normalized DW.
template< typename T >
DW< T > to_dw( mpfr_srcptr x )
{
   Big rest;
   const T hi = get< T >( x );
   mpfr_sub_d( rest, x, hi, MPFR_RNDN );
   return DW< T >( hi, get< T >( rest ) );
}

template< typename T >
void set( mpfr_ptr out, T hi, T lo )
{
   mpfr_set_d( out, hi, MPFR_RNDN );
   mpfr_add_d( out, out, lo, MPFR_RNDN );
}

template< typename T >
void set( mpfr_ptr out, const DW< T >& x )
{
   set( out, x.hi(), x.lo() );
}

template< typename T >
double rel_error( mpfr_srcptr exact, T hi, T lo )
{
   Big err;
   set( err, hi, lo );
   mpfr_sub( err, err, exact, MPFR_RNDN );
   mpfr_div( err, err, exact, MPFR_RNDN );
   return std::fabs( mpfr_get_d( err, MPFR_RNDN ) );
}

template< typename T >
double rel_error( mpfr_srcptr exact, const DW< T >& x )
{
   return rel_error( exact, x.hi(), x.lo() );
}

// Random values with a full-precision significand, random sign and exponent in [-20, 20], so
// float products and quotients of two stay normal.
struct Source
{
   std::mt19937_64 rng;
   gmp_randstate_t state;
   std::uniform_int_distribution< int > exponent{ -20, 20 };

   explicit Source( unsigned long seed ) : rng( seed )
   {
      gmp_randinit_mt( state );
      gmp_randseed_ui( state, seed );
   }
   ~Source() { gmp_randclear( state ); }

   void draw( mpfr_ptr x )
   {
      mpfr_urandom( x, state, MPFR_RNDN );
      mpfr_add_d( x, x, 0.5, MPFR_RNDN );
      mpfr_mul_2si( x, x, exponent( rng ), MPFR_RNDN );
      if( rng() & 1 )
         mpfr_neg( x, x, MPFR_RNDN );
   }

   // -x * (1 + 2^-k r) with k in [1, 100]: adding it to x cancels about k bits.
   void near_negation( mpfr_ptr out, mpfr_srcptr x )
   {
      const int k = std::uniform_int_distribution< int >( 1, 100 )( rng );
      mpfr_urandom( out, state, MPFR_RNDN );
      mpfr_mul_2si( out, out, -k, MPFR_RNDN );
      mpfr_add_ui( out, out, 1, MPFR_RNDN );
      mpfr_mul( out, out, x, MPFR_RNDN );
      mpfr_neg( out, out, MPFR_RNDN );
   }

   template< typename T >
   DW< T > dw()
   {
      Big x;
      draw( x );
      return to_dw< T >( x );
   }

   // A DW nearly cancelling `x`: x + result loses up to 100 bits.
   template< typename T >
   DW< T > dw_near_negation( const DW< T >& x )
   {
      Big a, b;
      set( a, x );
      near_negation( b, a );
      return to_dw< T >( b );
   }

   template< typename T >
   T fp()
   {
      return dw< T >().hi();
   }
};

#endif
