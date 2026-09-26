// DW<V> and XDW<V> with V a SIMD vector of float/double must give, in every lane, bit for bit what
// DW<lane> and XDW<lane> give on that lane's inputs. V is ext_vector_type on clang and vector_size on
// g++, as in MadGraph (clang mangles both kinds alike, so one translation unit can't mix them).

#include <cmath>
#include <limits>
#include <random>
#include <string>

#include "XDW.h"
#include "unit_test.h"

using namespace XDW_ARTH;

#if defined( __clang__ )
template< typename S, int N >
using vec = S __attribute__( ( ext_vector_type( N ) ) );
#else
template< typename S, int N >
struct vec_helper
{
   typedef S type __attribute__( ( vector_size( N * sizeof( S ) ) ) );
};
template< typename S, int N >
using vec = typename vec_helper< S, N >::type;
#endif

constexpr int TRIALS = 20000;

// Normalized DW with a random 2p-bit significand, sign and exponent in [-20, 20].
template< typename S >
static DW< S > random_dw( std::mt19937_64& rng )
{
   std::uniform_real_distribution< S > mantissa( S( 1 ), S( 2 ) ), unit( S( -1 ), S( 1 ) );
   std::uniform_int_distribution< int > exponent( -20, 20 );
   S h = std::ldexp( mantissa( rng ), exponent( rng ) );
   if( rng() & 1 )
      h = -h;
   const S l = std::ldexp( unit( rng ), std::ilogb( h ) - std::numeric_limits< S >::digits );
   const S hi = h + l;
   return DW< S >( hi, l - ( hi - h ) );
}

template< typename V, typename S = lane_t< V >, int N = lanes< V > >
static DW< V > pack( const DW< S > ( &x )[ N ] )
{
   V h, l;
   for( int i = 0; i < N; ++i ) {
      h[ i ] = x[ i ].hi();
      l[ i ] = x[ i ].lo();
   }
   return DW< V >( h, l );
}

// One trial: per-lane scalar inputs and the same values packed into vectors.
template< typename V >
struct Trial
{
   using S = lane_t< V >;
   static constexpr int N = lanes< V >;
   DW< S > xs[ N ], ys[ N ], us[ N ], vs[ N ];
   XDW< S > zs[ N ], ws[ N ];
   DW< V > x, y, u, v;
   XDW< V > z, w;

   // Every fourth trial, lane 0 of y shares x's hi (comparisons decided by lo); every third, the last
   // lane of x is +0 or -0.
   Trial( std::mt19937_64& rng, int t )
   {
      for( int i = 0; i < N; ++i ) {
         xs[ i ] = random_dw< S >( rng );
         ys[ i ] = random_dw< S >( rng );
         us[ i ] = random_dw< S >( rng );
         vs[ i ] = random_dw< S >( rng );
      }
      if( t % 4 == 0 )
         ys[ 0 ] = DW< S >( xs[ 0 ].hi(), xs[ 0 ].lo() * S( 0.5 ) );
      for( int i = 0; i < N; ++i ) {
         zs[ i ] = XDW< S >( xs[ i ], ys[ i ] );
         ws[ i ] = XDW< S >( us[ i ], vs[ i ] );
      }
      x = pack< V >( xs );
      y = pack< V >( ys );
      u = pack< V >( us );
      v = pack< V >( vs );
      z = XDW< V >( x, y );
      w = XDW< V >( u, v );
   }
};

// Runs `check` on TRIALS fresh trials (same seed for every group) and reports the differing lanes.
template< typename V, typename Check >
static void group( const std::string& what, Check check )
{
   unit::announce( what );
   std::mt19937_64 rng( 7 );
   long bad = 0, total = 0;
   for( int t = 0; t < TRIALS; ++t ) {
      const Trial< V > trial( rng, t );
      check( trial, t, bad, total );
   }
   unit::verdict( bad == 0, std::to_string( bad ) + " of " + std::to_string( total ) + " lane results differ" );
}

template< typename V, AddMode A, NormMode Nm >
static void check_modes( const Trial< V >& c, long& bad )
{
   const XDW< V > m = XDW< V >::template mul< A, Nm >( c.z, c.w );
   const XDW< V > d2 = XDW< V >::template div< DivMode::Div2, A, Nm >( c.z, c.w );
   const XDW< V > d3 = XDW< V >::template div< DivMode::Div3, A, Nm >( c.z, c.w );
   const DW< V > n = norm< V, A, Nm >( c.z );
   using S = lane_t< V >;
   for( int i = 0; i < Trial< V >::N; ++i ) {
      bad += !( m.lane( i ) == XDW< S >::template mul< A, Nm >( c.zs[ i ], c.ws[ i ] ) );
      bad += !( d2.lane( i ) == XDW< S >::template div< DivMode::Div2, A, Nm >( c.zs[ i ], c.ws[ i ] ) );
      bad += !( d3.lane( i ) == XDW< S >::template div< DivMode::Div3, A, Nm >( c.zs[ i ], c.ws[ i ] ) );
      bad += !( n.lane( i ) == norm< S, A, Nm >( c.zs[ i ] ) );
   }
}

template< typename V >
static void run( const char* name )
{
   using S = lane_t< V >;
   unit::section( std::string( name ) + ": every lane vs scalar DW/XDW, " + std::to_string( TRIALS ) + " trials" );

   group< V >( "DW + - * /, unary -, mixed with plain numbers, < <= > >= == != as lane masks",
               []( const Trial< V >& c, int, long& bad, long& total ) {
                  const DW< V > add = c.x + c.y, sub = c.x - c.y, mul = c.x * c.y, div = c.x / c.y, neg = -c.x;
                  const DW< V > mixed = ( c.x + 0.5 ) * 2 - 1.0 / c.y;
                  const auto lt = c.x < c.y, le = c.x <= c.y, gt = c.x > c.y, ge = c.x >= c.y, eq = c.x == c.y, ne = c.x != c.y;
                  for( int i = 0; i < lanes< V >; ++i ) {
                     const DW< S > a = c.xs[ i ], b = c.ys[ i ];
                     bad += !( add.lane( i ) == a + b ) + !( sub.lane( i ) == a - b ) + !( mul.lane( i ) == a * b )
                          + !( div.lane( i ) == a / b ) + !( neg.lane( i ) == -a ) + !( mixed.lane( i ) == ( a + 0.5 ) * 2 - 1.0 / b );
                     bad += ( ( lt[ i ] != 0 ) != ( a < b ) ) + ( ( le[ i ] != 0 ) != ( a <= b ) ) + ( ( gt[ i ] != 0 ) != ( a > b ) )
                          + ( ( ge[ i ] != 0 ) != ( a >= b ) ) + ( ( eq[ i ] != 0 ) != ( a == b ) ) + ( ( ne[ i ] != 0 ) != ( a != b ) );
                  }
                  total += 12 * lanes< V >;
               } );

   group< V >( "sqrt, abs, signbit, min, max, select (DW and XDW), with +0 and -0 lanes",
               []( const Trial< V >& c, int t, long& bad, long& total ) {
                  DW< S > ps[ lanes< V > ];
                  for( int i = 0; i < lanes< V >; ++i ) ps[ i ] = c.xs[ i ];
                  if( t % 3 == 0 )
                     ps[ lanes< V > - 1 ] = t % 2 ? DW< S >( S( 0 ), S( 0 ) ) : DW< S >( -S( 0 ), -S( 0 ) );
                  const DW< V > p = pack< V >( ps );
                  const DW< V > sq = sqrt( abs( p ) ), ab = abs( p ), mn = min( p, c.y ), mx = max( p, 0.5 ), sl = select( p < c.y, p, c.y );
                  const auto sb = signbit( p );
                  const XDW< V > zsel = select( p < c.y, c.z, c.w );
                  for( int i = 0; i < lanes< V >; ++i ) {
                     const DW< S > a = ps[ i ], b = c.ys[ i ];
                     bad += !( sq.lane( i ) == sqrt( abs( a ) ) ) + !( ab.lane( i ) == abs( a ) ) + !( mn.lane( i ) == min( a, b ) )
                          + !( mx.lane( i ) == max( a, 0.5 ) ) + !( sl.lane( i ) == select( a < b, a, b ) )
                          + ( ( sb[ i ] != 0 ) != signbit( a ) ) + !( zsel.lane( i ) == select( a < b, c.zs[ i ], c.ws[ i ] ) );
                     // abs(-0) is +0: == treats -0 and +0 as equal, so check the sign too.
                     bad += signbit( ab.lane( i ) );
                  }
                  total += 8 * lanes< V >;
               } );

   group< V >( "XDW default operators, conj, real, imag, ==, and with a DW: * / + -",
               []( const Trial< V >& c, int, long& bad, long& total ) {
                  const XDW< V > ops = c.z * c.w / c.w + c.z - c.w;
                  const XDW< V > zx = c.z * c.x, xz = c.x * c.z, zdx = c.z / c.x, zpx = c.z + c.x, xmz = c.x - c.z, zh = c.z * 0.5 + 1;
                  const XDW< V > cz = conj( c.z );
                  const auto zeq = c.z == c.w;
                  for( int i = 0; i < lanes< V >; ++i ) {
                     const XDW< S > a = c.zs[ i ], b = c.ws[ i ];
                     const DW< S > r = c.xs[ i ];
                     bad += !( ops.lane( i ) == a * b / b + a - b );
                     bad += !( zx.lane( i ) == a * r ) + !( xz.lane( i ) == r * a ) + !( zdx.lane( i ) == a / r ) + !( zpx.lane( i ) == a + r )
                          + !( xmz.lane( i ) == r - a ) + !( zh.lane( i ) == a * 0.5 + 1 );
                     bad += !( cz.lane( i ) == conj( a ) ) + !( real( c.z ).lane( i ) == real( a ) ) + !( imag( c.z ).lane( i ) == imag( a ) );
                     bad += ( ( zeq[ i ] != 0 ) != ( a == b ) );
                  }
                  total += 11 * lanes< V >;
               } );

   group< V >( "mul and norm in all 6 modes, div in all 12 modes",
               []( const Trial< V >& c, int, long& bad, long& total ) {
                  check_modes< V, AddMode::Madd, NormMode::Normalized >( c, bad );
                  check_modes< V, AddMode::Madd, NormMode::Unnormalized >( c, bad );
                  check_modes< V, AddMode::Accurate, NormMode::Normalized >( c, bad );
                  check_modes< V, AddMode::Accurate, NormMode::Unnormalized >( c, bad );
                  check_modes< V, AddMode::Sloppy, NormMode::Normalized >( c, bad );
                  check_modes< V, AddMode::Sloppy, NormMode::Unnormalized >( c, bad );
                  total += 24 * lanes< V >;
               } );

   group< V >( "broadcasts: a plain number, a scalar DW and a scalar XDW in every lane",
               []( const Trial< V >& c, int, long& bad, long& total ) {
                  const DW< V > bx( c.xs[ 0 ] ), b01( 0.1 );
                  const XDW< V > bz( c.zs[ 0 ] ), b12( 0.1, 2 );
                  for( int i = 0; i < lanes< V >; ++i )
                     bad += !( bx.lane( i ) == c.xs[ 0 ] ) + !( b01.lane( i ) == DW< S >( 0.1 ) ) + !( bz.lane( i ) == c.zs[ 0 ] )
                          + !( b12.lane( i ) == XDW< S >( 0.1, 2 ) );
                  total += 4 * lanes< V >;
               } );

   group< V >( "mixed operands: scalar DW/XDW or a plain number next to a vector one, DW / XDW",
               []( const Trial< V >& c, int, long& bad, long& total ) {
                  const DW< S > rs = c.us[ 0 ];
                  const XDW< S > cs = c.ws[ 0 ];
                  const XDW< V > m1 = c.z + cs, m2 = cs - c.z, m3 = c.z * cs, m4 = cs / c.z;
                  const XDW< V > m5 = c.x + cs, m6 = cs - c.x, m7 = c.x * cs, m8 = cs / c.x;
                  const XDW< V > m9 = rs + c.z, m10 = c.z - rs, m11 = rs * c.z, m12 = c.z / rs;
                  const XDW< V > m13 = c.z * 0.5 + 2, m14 = 3 - c.z / 0.25, m15 = c.x / c.z;
                  const DW< V > m16 = c.x * rs - rs / c.x;
                  for( int i = 0; i < lanes< V >; ++i ) {
                     const XDW< S > a = c.zs[ i ];
                     const DW< S > r = c.xs[ i ];
                     bad += !( m1.lane( i ) == a + cs ) + !( m2.lane( i ) == cs - a ) + !( m3.lane( i ) == a * cs ) + !( m4.lane( i ) == cs / a );
                     bad += !( m5.lane( i ) == XDW< S >( r ) + cs ) + !( m6.lane( i ) == cs - r ) + !( m7.lane( i ) == r * cs ) + !( m8.lane( i ) == cs / r );
                     bad += !( m9.lane( i ) == rs + a ) + !( m10.lane( i ) == a - rs ) + !( m11.lane( i ) == rs * a ) + !( m12.lane( i ) == a / rs );
                     bad += !( m13.lane( i ) == a * DW< S >( 0.5 ) + DW< S >( 2 ) ) + !( m14.lane( i ) == DW< S >( 3 ) - a / DW< S >( 0.25 ) )
                          + !( m15.lane( i ) == r / a ) + !( m16.lane( i ) == r * rs - rs / r );
                  }
                  total += 16 * lanes< V >;
               } );
}

int main()
{
   run< vec< double, 2 > >( "double x2" );
   run< vec< double, 4 > >( "double x4" );
   run< vec< double, 8 > >( "double x8" );
   run< vec< float, 4 > >( "float x4" );
   run< vec< float, 8 > >( "float x8" );
   run< vec< float, 16 > >( "float x16" );
   return unit::finish( "test_simd" );
}
