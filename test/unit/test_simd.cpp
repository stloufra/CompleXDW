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

// Counts compared lane results and those that differ.
struct Tally
{
   long bad = 0, total = 0;

   void operator()( bool same )
   {
      ++total;
      bad += !same;
   }
};

// Lane i of a vector result equals the scalar result.
template< typename Vector, typename Scalar >
static bool same( const Vector& vector_result, int i, const Scalar& scalar_result )
{
   return vector_result.lane( i ) == scalar_result;
}

// Lane i of a comparison mask agrees with the scalar comparison.
template< typename Mask >
static bool agrees( const Mask& mask, int i, bool scalar_result )
{
   return ( mask[ i ] != 0 ) == scalar_result;
}

// Runs `check` on TRIALS fresh trials (same seed for every group) and reports the differing lanes.
template< typename V, typename Check >
static void group( const std::string& name, const std::string& note, Check check )
{
   unit::announce( name, note );
   std::mt19937_64 rng( 7 );
   Tally tally;
   for( int t = 0; t < TRIALS; ++t ) {
      const Trial< V > trial( rng, t );
      check( trial, t, tally );
   }
   unit::verdict( tally.bad == 0, std::to_string( tally.bad ) + " / " + std::to_string( tally.total ) + " lanes differ" );
}

// mul, div (Div2, Div3) and norm in one (Add, Norm) mode.
template< typename V, AddMode A, NormMode Nm >
static void check_modes( const Trial< V >& c, Tally& tally )
{
   using S = lane_t< V >;
   using CV = XDW< V >;
   using CS = XDW< S >;

   // Vector results.
   const CV m = CV::template mul< A, Nm >( c.z, c.w );
   const CV d2 = CV::template div< DivMode::Div2, A, Nm >( c.z, c.w );
   const CV d3 = CV::template div< DivMode::Div3, A, Nm >( c.z, c.w );
   const DW< V > n = norm< V, A, Nm >( c.z );

   // Each lane against the scalar XDW.
   for( int i = 0; i < lanes< V >; ++i ) {
      const CS a = c.zs[ i ], b = c.ws[ i ];
      tally( same( m, i, CS::template mul< A, Nm >( a, b ) ) );
      tally( same( d2, i, CS::template div< DivMode::Div2, A, Nm >( a, b ) ) );
      tally( same( d3, i, CS::template div< DivMode::Div3, A, Nm >( a, b ) ) );
      tally( same( n, i, norm< S, A, Nm >( a ) ) );
   }
}

template< typename V >
static void run( const char* name )
{
   using S = lane_t< V >;
   unit::section( name, std::to_string( TRIALS ) + " trials per check, every lane vs scalar DW/XDW" );

   group< V >( "DW arithmetic, comparisons", "+ - * /, -x, numbers, masks", []( const Trial< V >& c, int, Tally& tally ) {
      // Vector results, all lanes at once.
      const DW< V > add = c.x + c.y, sub = c.x - c.y, mul = c.x * c.y, div = c.x / c.y, neg = -c.x;
      const DW< V > mixed = ( c.x + 0.5 ) * 2 - 1.0 / c.y;  // plain numbers go to every lane
      const auto lt = c.x < c.y, le = c.x <= c.y, gt = c.x > c.y, ge = c.x >= c.y, eq = c.x == c.y, ne = c.x != c.y;

      for( int i = 0; i < lanes< V >; ++i ) {
         const DW< S > a = c.xs[ i ], b = c.ys[ i ];

         // Arithmetic.
         tally( same( add, i, a + b ) );
         tally( same( sub, i, a - b ) );
         tally( same( mul, i, a * b ) );
         tally( same( div, i, a / b ) );
         tally( same( neg, i, -a ) );
         tally( same( mixed, i, ( a + 0.5 ) * 2 - 1.0 / b ) );

         // Comparisons: mask lane vs scalar bool.
         tally( agrees( lt, i, a < b ) );
         tally( agrees( le, i, a <= b ) );
         tally( agrees( gt, i, a > b ) );
         tally( agrees( ge, i, a >= b ) );
         tally( agrees( eq, i, a == b ) );
         tally( agrees( ne, i, a != b ) );
      }
   } );

   group< V >( "DW, XDW functions, +-0 lanes", "sqrt abs signbit min max select", []( const Trial< V >& c, int t, Tally& tally ) {
      // x, with +0 or -0 in the last lane every third trial.
      DW< S > ps[ lanes< V > ];
      for( int i = 0; i < lanes< V >; ++i )
         ps[ i ] = c.xs[ i ];
      if( t % 3 == 0 )
         ps[ lanes< V > - 1 ] = t % 2 ? DW< S >( S( 0 ), S( 0 ) ) : DW< S >( -S( 0 ), -S( 0 ) );
      const DW< V > p = pack< V >( ps );

      // Vector results.
      const DW< V > sq = sqrt( abs( p ) ), ab = abs( p ), mn = min( p, c.y ), mx = max( p, 0.5 );
      const DW< V > sel = select( p < c.y, p, c.y );
      const XDW< V > zsel = select( p < c.y, c.z, c.w );
      const auto sb = signbit( p );

      for( int i = 0; i < lanes< V >; ++i ) {
         const DW< S > a = ps[ i ], b = c.ys[ i ];
         tally( same( sq, i, sqrt( abs( a ) ) ) );
         tally( same( ab, i, abs( a ) ) );
         tally( !signbit( ab.lane( i ) ) );  // abs(-0) is +0; == alone can't tell -0 from +0
         tally( same( mn, i, min( a, b ) ) );
         tally( same( mx, i, max( a, 0.5 ) ) );
         tally( same( sel, i, select( a < b, a, b ) ) );
         tally( same( zsel, i, select( a < b, c.zs[ i ], c.ws[ i ] ) ) );
         tally( agrees( sb, i, signbit( a ) ) );
      }
   } );

   group< V >( "XDW operators", "+-*/, conj, real, imag, ==, DW", []( const Trial< V >& c, int, Tally& tally ) {
      // Vector results: complex z, w and the real DW x.
      const XDW< V > ops = c.z * c.w / c.w + c.z - c.w;
      const XDW< V > zx = c.z * c.x, xz = c.x * c.z, zdx = c.z / c.x, zpx = c.z + c.x, xmz = c.x - c.z;
      const XDW< V > zh = c.z * 0.5 + 1;
      const XDW< V > cz = conj( c.z );
      const auto zeq = c.z == c.w;

      for( int i = 0; i < lanes< V >; ++i ) {
         const XDW< S > a = c.zs[ i ], b = c.ws[ i ];
         const DW< S > r = c.xs[ i ];

         // Complex with complex.
         tally( same( ops, i, a * b / b + a - b ) );
         tally( agrees( zeq, i, a == b ) );

         // Complex with a real DW and plain numbers.
         tally( same( zx, i, a * r ) );
         tally( same( xz, i, r * a ) );
         tally( same( zdx, i, a / r ) );
         tally( same( zpx, i, a + r ) );
         tally( same( xmz, i, r - a ) );
         tally( same( zh, i, a * 0.5 + 1 ) );

         // Parts.
         tally( same( cz, i, conj( a ) ) );
         tally( same( real( c.z ), i, real( a ) ) );
         tally( same( imag( c.z ), i, imag( a ) ) );
      }
   } );

   group< V >( "XDW modes", "6 mul, 12 div, 6 norm", []( const Trial< V >& c, int, Tally& tally ) {
      check_modes< V, AddMode::Madd, NormMode::Normalized >( c, tally );
      check_modes< V, AddMode::Madd, NormMode::Unnormalized >( c, tally );
      check_modes< V, AddMode::Accurate, NormMode::Normalized >( c, tally );
      check_modes< V, AddMode::Accurate, NormMode::Unnormalized >( c, tally );
      check_modes< V, AddMode::Sloppy, NormMode::Normalized >( c, tally );
      check_modes< V, AddMode::Sloppy, NormMode::Unnormalized >( c, tally );
   } );

   group< V >( "broadcasts", "number, scalar DW, scalar XDW", []( const Trial< V >& c, int, Tally& tally ) {
      // One value copied into every lane.
      const DW< V > from_dw( c.xs[ 0 ] ), from_number( 0.1 );
      const XDW< V > from_xdw( c.zs[ 0 ] ), from_numbers( 0.1, 2 );

      for( int i = 0; i < lanes< V >; ++i ) {
         tally( same( from_dw, i, c.xs[ 0 ] ) );
         tally( same( from_number, i, DW< S >( 0.1 ) ) );
         tally( same( from_xdw, i, c.zs[ 0 ] ) );
         tally( same( from_numbers, i, XDW< S >( 0.1, 2 ) ) );
      }
   } );

   group< V >( "mixed operands", "scalar next to vector, DW / XDW", []( const Trial< V >& c, int, Tally& tally ) {
      // A scalar real rs and complex cs next to the vector z (complex) and x (real).
      const DW< S > rs = c.us[ 0 ];
      const XDW< S > cs = c.ws[ 0 ];
      const XDW< V > z_cs[] = { c.z + cs, cs - c.z, c.z * cs, cs / c.z };
      const XDW< V > x_cs[] = { c.x + cs, cs - c.x, c.x * cs, cs / c.x };
      const XDW< V > z_rs[] = { rs + c.z, c.z - rs, rs * c.z, c.z / rs };
      const XDW< V > z_numbers = c.z * 0.5 + 2, numbers_z = 3 - c.z / 0.25, x_over_z = c.x / c.z;
      const DW< V > x_rs = c.x * rs - rs / c.x;

      for( int i = 0; i < lanes< V >; ++i ) {
         const XDW< S > a = c.zs[ i ];
         const DW< S > r = c.xs[ i ];

         // Vector XDW with a scalar XDW.
         tally( same( z_cs[ 0 ], i, a + cs ) );
         tally( same( z_cs[ 1 ], i, cs - a ) );
         tally( same( z_cs[ 2 ], i, a * cs ) );
         tally( same( z_cs[ 3 ], i, cs / a ) );

         // Vector DW with a scalar XDW.
         tally( same( x_cs[ 0 ], i, XDW< S >( r ) + cs ) );
         tally( same( x_cs[ 1 ], i, cs - r ) );
         tally( same( x_cs[ 2 ], i, r * cs ) );
         tally( same( x_cs[ 3 ], i, cs / r ) );

         // Vector XDW with a scalar DW.
         tally( same( z_rs[ 0 ], i, rs + a ) );
         tally( same( z_rs[ 1 ], i, a - rs ) );
         tally( same( z_rs[ 2 ], i, rs * a ) );
         tally( same( z_rs[ 3 ], i, a / rs ) );

         // Plain numbers, DW / XDW, vector DW with a scalar DW.
         tally( same( z_numbers, i, a * DW< S >( 0.5 ) + DW< S >( 2 ) ) );
         tally( same( numbers_z, i, DW< S >( 3 ) - a / DW< S >( 0.25 ) ) );
         tally( same( x_over_z, i, r / a ) );
         tally( same( x_rs, i, r * rs - rs / r ) );
      }
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
