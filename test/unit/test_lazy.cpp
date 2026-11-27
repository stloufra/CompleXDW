// Lazy products under NormMode::Unnormalized: sum_k x_k*y_k + z fuses with unnormalized products,
// anything else between the products and the add evaluates eagerly.

#define XDW_NORM_MODE U

#include <algorithm>
#include <cmath>
#include <string>
#include <type_traits>

#include "mpfr_support.h"
#include "unit_test.h"

using namespace XDW_ARTH;

constexpr int SAMPLES_LAZY = 50000;
constexpr int TERMS = 4;

static_assert( kLazyProducts );
static_assert( std::is_same_v< decltype( DW< double >() * DW< double >() ), ProductSum< DW< double >, 1 > > );
static_assert( std::is_same_v< decltype( XDW< float >() * XDW< float >() - XDW< float >() * XDW< float >() ),
                               ProductSum< XDW< float >, 2 > > );
static_assert( std::is_same_v< decltype( XDW< double >() * XDW< double >() + XDW< double >() ), XDW< double > > );

template< typename T >
static XDW< T > random_xdw( Source& src )
{
   return XDW< T >( src.dw< T >(), src.dw< T >() );
}

template< typename T >
static DW< T > madd_unnorm( const DW< T >& x, const DW< T >& y, const DW< T >& z )
{
   T ph, pl, h, l;
   DWTimesDW2Unnorm( x.hi(), x.lo(), y.hi(), y.lo(), &ph, &pl );
   MaddDWPlusDW( ph, pl, z.hi(), z.lo(), &h, &l );
   return DW< T >( h, l );
}

// (sum |terms|) / |sum terms|.
static double conditioning( mpfr_srcptr sum, mpfr_srcptr magnitude )
{
   Big k;
   mpfr_div( k, magnitude, sum, MPFR_RNDN );
   return std::fabs( mpfr_get_d( k, MPFR_RNDN ) );
}

// out += sign * a * b, magnitude += |a * b|.
static void accumulate( mpfr_ptr out, mpfr_ptr magnitude, mpfr_srcptr a, mpfr_srcptr b, int sign )
{
   Big p;
   mpfr_mul( p, a, b, MPFR_RNDN );
   if( sign < 0 )
      mpfr_neg( p, p, MPFR_RNDN );
   mpfr_add( out, out, p, MPFR_RNDN );
   mpfr_abs( p, p, MPFR_RNDN );
   mpfr_add( magnitude, magnitude, p, MPFR_RNDN );
}

template< typename T >
static void test_dw( Source& src )
{
   unit::announce( "DW x*y + z", "DWTimesDW2Unnorm then Madd, bitwise" );
   bool same = true;
   for( int i = 0; i < SAMPLES_LAZY; ++i ) {
      const DW< T > x = src.dw< T >(), y = src.dw< T >(), z = src.dw< T >();
      const DW< T > expected = madd_unnorm( x, y, z );
      DW< T > acc = z;
      acc += x * y;
      same &= x * y + z == expected && z + x * y == expected && acc == expected;
   }
   unit::verdict( same );

   unit::announce( "DW x*y + z", "half cancelling" );
   Big a, b, c, ref, magnitude;
   double worst = 0;
   for( int i = 0; i < SAMPLES_LAZY; ++i ) {
      const DW< T > x = src.dw< T >(), y = src.dw< T >();
      set( a, x );
      set( b, y );
      mpfr_mul( ref, a, b, MPFR_RNDN );
      const DW< T > z = i % 2 ? src.dw_near_negation( to_dw< T >( ref ) ) : src.dw< T >();
      set( c, z );
      mpfr_set( ref, c, MPFR_RNDN );
      mpfr_abs( magnitude, c, MPFR_RNDN );
      accumulate( ref, magnitude, a, b, 1 );
      if( !mpfr_zero_p( ref ) )
         worst = std::max( worst, rel_error( ref, x * y + z ) / conditioning( ref, magnitude ) );
   }
   unit::verdict_bound( worst / U2< T >, 8, "K u^2" );

   unit::announce( "DW sum_k x_k*y_k + z", std::to_string( TERMS ) + " products, fused" );
   double worst_fused = 0, worst_eager = 0;
   for( int i = 0; i < SAMPLES_LAZY; ++i ) {
      DW< T > x[ TERMS ], y[ TERMS ];
      const DW< T > z = src.dw< T >();
      set( ref, z );
      mpfr_abs( magnitude, ref, MPFR_RNDN );
      for( int k = 0; k < TERMS; ++k ) {
         x[ k ] = src.dw< T >();
         y[ k ] = src.dw< T >();
         set( a, x[ k ] );
         set( b, y[ k ] );
         accumulate( ref, magnitude, a, b, 1 );
      }
      const DW< T > fused = x[ 0 ] * y[ 0 ] + x[ 1 ] * y[ 1 ] + x[ 2 ] * y[ 2 ] + x[ 3 ] * y[ 3 ] + z;
      DW< T > eager = z;
      for( int k = 0; k < TERMS; ++k )
         eager = eager + DW< T >::mul( x[ k ], y[ k ] );
      const double kappa = conditioning( ref, magnitude );
      worst_fused = std::max( worst_fused, rel_error( ref, fused ) / kappa );
      worst_eager = std::max( worst_eager, rel_error( ref, eager ) / kappa );
   }
   unit::info( worst_fused / U2< T >, "K u^2" );
   unit::announce( "", "eager, DWTimesDW2 per product" );
   unit::info( worst_eager / U2< T >, "K u^2" );
}

template< typename T >
static void test_xdw( Source& src )
{
   unit::announce( "XDW x*y + z", "equals mul<Add, U> + z bitwise" );
   bool same = true;
   for( int i = 0; i < SAMPLES_LAZY; ++i ) {
      const XDW< T > x = random_xdw< T >( src ), y = random_xdw< T >( src ), z = random_xdw< T >( src );
      const XDW< T > expected = XDW< T >::add( XDW< T >::mul( x, y ), z );
      same &= x * y + z == expected && z + x * y == expected;
   }
   unit::verdict( same );

   unit::announce( "XDW sum_k x_k*y_k + z", std::to_string( TERMS ) + " products, fused" );
   Big a, b, c, d, re, im, re_mag, im_mag;
   double worst_fused = 0, worst_eager = 0;
   for( int i = 0; i < SAMPLES_LAZY; ++i ) {
      XDW< T > x[ TERMS ], y[ TERMS ];
      const XDW< T > z = random_xdw< T >( src );
      set( re, real( z ) );
      set( im, imag( z ) );
      mpfr_abs( re_mag, re, MPFR_RNDN );
      mpfr_abs( im_mag, im, MPFR_RNDN );
      for( int k = 0; k < TERMS; ++k ) {
         x[ k ] = random_xdw< T >( src );
         y[ k ] = random_xdw< T >( src );
         set( a, real( x[ k ] ) );
         set( b, imag( x[ k ] ) );
         set( c, real( y[ k ] ) );
         set( d, imag( y[ k ] ) );
         accumulate( re, re_mag, a, c, 1 );
         accumulate( re, re_mag, b, d, -1 );
         accumulate( im, im_mag, a, d, 1 );
         accumulate( im, im_mag, b, c, 1 );
      }
      const XDW< T > fused = x[ 0 ] * y[ 0 ] + x[ 1 ] * y[ 1 ] + x[ 2 ] * y[ 2 ] + x[ 3 ] * y[ 3 ] + z;
      XDW< T > eager = z;
      for( int k = 0; k < TERMS; ++k )
         eager = XDW< T >::add( eager, XDW< T >::mul( x[ k ], y[ k ] ) );
      const double re_kappa = conditioning( re, re_mag ), im_kappa = conditioning( im, im_mag );
      worst_fused = std::max( { worst_fused, rel_error( re, real( fused ) ) / re_kappa, rel_error( im, imag( fused ) ) / im_kappa } );
      worst_eager = std::max( { worst_eager, rel_error( re, real( eager ) ) / re_kappa, rel_error( im, imag( eager ) ) / im_kappa } );
   }
   unit::info( worst_fused / U2< T >, "K u^2" );
   unit::announce( "", "eager, normalized per product" );
   unit::info( worst_eager / U2< T >, "K u^2" );
}

template< typename T >
static T generic_root( T x )
{
   return sqrt( x );
}

template< typename T >
static T generic_sum( T a, T b )
{
   T acc = 0;
   acc += a;
   acc = acc + b;
   return acc;
}

// Minus only negates factors, so it stays fused; any other operation evaluates the product first.
template< typename T >
static void test_pattern( Source& src )
{
   unit::announce( "minus between products", "negated factor, still fused" );
   bool minus = true;
   for( int i = 0; i < SAMPLES_LAZY; ++i ) {
      const XDW< T > x = random_xdw< T >( src ), y = random_xdw< T >( src ), v = random_xdw< T >( src ),
                     w = random_xdw< T >( src ), z = random_xdw< T >( src );
      XDW< T > acc = z;
      acc -= x * y;
      minus &= x * y - v * w + z == x * y + ( -v ) * w + z && z - x * y == ( -x ) * y + z
            && x * y - z == x * y + ( -z ) && acc == z - x * y;
   }
   unit::verdict( minus );

   unit::announce( "other operation in between", "eager product, then +" );
   bool eager = true;
   for( int i = 0; i < SAMPLES_LAZY; ++i ) {
      const XDW< T > x = random_xdw< T >( src ), y = random_xdw< T >( src ), w = random_xdw< T >( src ),
                     z = random_xdw< T >( src );
      const XDW< T > p = XDW< T >::mul( x, y );
      const XDW< T > stored = x * y;
      const DW< T > r = src.dw< T >(), s = src.dw< T >(), t = src.dw< T >();
      const DW< T > q = DW< T >::mul( r, s );
      eager &= stored + z == p + z && -( x * y ) + z == -p + z && ( x * y ) / w + z == p / w + z
            && conj( x * y ) + z == conj( p ) + z && ( x * y ) * 2.0 + z == p * 2.0 + z
            && ( x * y ) * t + z == p * t + z && ( r * s ) / t + t == q / t + t && sqrt( abs( r * s ) ) + t == sqrt( abs( q ) ) + t;
   }
   unit::verdict( eager );

   unit::announce( "named product", "eager, also after +=, *=" );
   bool named = true;
   for( int i = 0; i < SAMPLES_LAZY; ++i ) {
      const DW< T > r = src.dw< T >(), s = src.dw< T >(), t = src.dw< T >(), u = src.dw< T >();
      auto p = r * s;
      named &= p + t == DW< T >::mul( r, s ) + t;
      p += t;
      named &= p + u == DW< T >( p ) + u;
      p *= u;
      named &= p + t == DW< T >( p ) + t;
   }
   unit::verdict( named );

   unit::announce( "generic T f(T)", "T = ProductSum holds its value" );
   bool generic = true;
   for( int i = 0; i < SAMPLES_LAZY; ++i ) {
      const DW< T > r = src.dw< T >(), s = src.dw< T >(), t = src.dw< T >();
      const XDW< T > x = random_xdw< T >( src ), y = random_xdw< T >( src ), z = random_xdw< T >( src );
      const DW< T > sum = generic_sum( r * s, t * t );
      generic &= generic_root( r * r ) == sqrt( DW< T >::mul( r, r ) ) && generic_sum( r * s, t * t ) + t == sum + t
              && generic_sum( x * y, z * z ) + z == XDW< T >( generic_sum( x * y, z * z ) ) + z;
   }
   unit::verdict( generic );
}

template< typename T >
static void run( const char* type, unsigned long seed )
{
   Source src( seed );
   const std::string t = std::string( "<" ) + type + ">: ";
   unit::section( std::string( "DW" ) + t + "lazy products", std::to_string( SAMPLES_LAZY ) + " samples per check, relative error vs MPFR" );
   test_dw< T >( src );
   unit::section( std::string( "XDW" ) + t + "lazy products", std::to_string( SAMPLES_LAZY ) + " samples per check, relative error per component vs MPFR" );
   test_xdw< T >( src );
   unit::section( std::string( "XDW" ) + t + "pattern", std::to_string( SAMPLES_LAZY ) + " samples per check" );
   test_pattern< T >( src );
}

int main()
{
   mpfr_set_default_prec( MPFR_BITS );
   describe_inputs();
   run< double >( "double", 61 );
   run< float >( "float", 62 );
   mpfr_free_cache();
   return unit::finish( "test_lazy" );
}
