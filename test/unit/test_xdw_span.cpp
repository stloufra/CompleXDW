// XDWSpan bulk mul/div (double and float): bit-identical to per-element XDW in every mode, in place
// too, and mismatched sizes are rejected.

#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "XDW.h"
#include "XDWSpan.h"
#include "unit_test.h"

using namespace XDW_ARTH;

constexpr std::size_t N = 1000;

template< typename T >
struct SoA
{
   std::vector< T > re_h, re_l, im_h, im_l;

   explicit SoA( std::size_t n ) : re_h( n ), re_l( n ), im_h( n ), im_l( n ) {}

   XDWSpan< T > span() { return { re_h.data(), re_l.data(), im_h.data(), im_l.data(), re_h.size() }; }
};

// Normalized DW with a random 2p-bit significand, sign and exponent in [-20, 20].
template< typename T >
static DW< T > random_dw( std::mt19937_64& rng )
{
   std::uniform_real_distribution< T > mantissa( T( 1 ), T( 2 ) ), unit( T( -1 ), T( 1 ) );
   std::uniform_int_distribution< int > exponent( -20, 20 );
   T h = std::ldexp( mantissa( rng ), exponent( rng ) );
   if( rng() & 1 )
      h = -h;
   const T l = std::ldexp( unit( rng ), std::ilogb( h ) - std::numeric_limits< T >::digits );
   const T hi = h + l;
   return DW< T >( hi, l - ( hi - h ) );
}

template< typename T, AddMode Add, NormMode Norm >
static bool mul_matches( SoA< T >& a, SoA< T >& b )
{
   SoA< T > out( N );
   mul< Add, Norm >( out.span(), a.span(), b.span() );
   bool same = true;
   for( std::size_t i = 0; i < N; ++i )
      same &= out.span().load( i ) == XDW< T >::template mul< Add, Norm >( a.span().load( i ), b.span().load( i ) );
   return same;
}

template< typename T, DivMode Div, AddMode Add, NormMode Norm >
static bool div_matches( SoA< T >& a, SoA< T >& b )
{
   SoA< T > out( N );
   div< Div, Add, Norm >( out.span(), a.span(), b.span() );
   bool same = true;
   for( std::size_t i = 0; i < N; ++i )
      same &= out.span().load( i ) == XDW< T >::template div< Div, Add, Norm >( a.span().load( i ), b.span().load( i ) );
   return same;
}

template< typename T, AddMode Add, NormMode Norm >
static bool modes_match( SoA< T >& a, SoA< T >& b )
{
   return mul_matches< T, Add, Norm >( a, b ) && div_matches< T, DivMode::Div2, Add, Norm >( a, b ) && div_matches< T, DivMode::Div3, Add, Norm >( a, b );
}

template< typename T >
static void run( const char* type, unsigned long seed )
{
   unit::section( std::string( "XDWSpan<" ) + type + ">, " + std::to_string( N ) + " elements" );
   std::mt19937_64 rng( seed );
   SoA< T > a( N ), b( N );
   for( std::size_t i = 0; i < N; ++i ) {
      a.span().store( i, XDW< T >( random_dw< T >( rng ), random_dw< T >( rng ) ) );
      b.span().store( i, XDW< T >( random_dw< T >( rng ), random_dw< T >( rng ) ) );
   }

   unit::announce( "store then load returns the same XDW" );
   bool round_trip = true;
   for( std::size_t i = 0; i < N; ++i ) {
      const XDW< T > z( random_dw< T >( rng ), random_dw< T >( rng ) );
      SoA< T > one( 1 );
      one.span().store( 0, z );
      round_trip &= one.span().load( 0 ) == z;
   }
   unit::verdict( round_trip );

   unit::announce( "mul and div, all 6 + 12 modes, equal per-element XDW bit for bit" );
   unit::verdict( modes_match< T, AddMode::Madd, NormMode::Normalized >( a, b ) && modes_match< T, AddMode::Madd, NormMode::Unnormalized >( a, b )
                  && modes_match< T, AddMode::Accurate, NormMode::Normalized >( a, b ) && modes_match< T, AddMode::Accurate, NormMode::Unnormalized >( a, b )
                  && modes_match< T, AddMode::Sloppy, NormMode::Normalized >( a, b ) && modes_match< T, AddMode::Sloppy, NormMode::Unnormalized >( a, b ) );

   unit::announce( "mul and div with default modes equal operator* and operator/" );
   SoA< T > product( N ), quotient( N );
   mul( product.span(), a.span(), b.span() );
   div( quotient.span(), a.span(), b.span() );
   bool defaults = true;
   for( std::size_t i = 0; i < N; ++i )
      defaults &= product.span().load( i ) == a.span().load( i ) * b.span().load( i )
               && quotient.span().load( i ) == a.span().load( i ) / b.span().load( i );
   unit::verdict( defaults );

   unit::announce( "in place: out may be one of the inputs" );
   SoA< T > in_place = a;
   mul( in_place.span(), in_place.span(), b.span() );
   div( in_place.span(), in_place.span(), b.span() );
   bool same = true;
   for( std::size_t i = 0; i < N; ++i )
      same &= in_place.span().load( i ) == product.span().load( i ) / b.span().load( i );
   unit::verdict( same );

   unit::announce( "operands of different sizes throw std::invalid_argument" );
   SoA< T > shorter( N - 1 );
   bool threw = false;
   try {
      mul( shorter.span(), a.span(), b.span() );
   }
   catch( const std::invalid_argument& ) {
      threw = true;
   }
   unit::verdict( threw );
}

int main()
{
   run< double >( "double", 61 );
   run< float >( "float", 62 );
   return unit::finish( "test_xdw_span" );
}
