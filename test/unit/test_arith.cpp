// Every XDW_ARTH building block against MPFR, for double and float: the error-free transforms
// must be exact, the DW algorithms within their documented relative error bounds.

#include <algorithm>
#include <cmath>
#include <string>

#include "mpfr_support.h"
#include "unit_test.h"

using namespace XDW_ARTH;

enum class Pairs { Random, Cancelling, Positive };

// Second operand of the DW-FP algorithms: a plain T, i.e. a DW with a zero low word.
enum class Second { DW, FP };

// Every other sample of Cancelling makes y nearly -x.
template< typename T >
static DW< T > partner( Source& src, const DW< T >& x, Pairs pairs, int i )
{
   if( pairs == Pairs::Positive )
      return abs( src.dw< T >() );
   if( pairs == Pairs::Cancelling && i % 2 )
      return src.dw_near_negation( x );
   return src.dw< T >();
}

// Worst relative error / u^2 of f(x, y) -> (h, l) against exact(x, y).
template< typename T, typename F, typename E >
static double worst_binary( Source& src, Pairs pairs, F f, E exact, Second second = Second::DW )
{
   Big a, b, ref;
   double worst = 0;
   for( int i = 0; i < SAMPLES; ++i ) {
      const DW< T > x = pairs == Pairs::Positive ? abs( src.dw< T >() ) : src.dw< T >();
      DW< T > y = partner( src, x, pairs, i );
      if( second == Second::FP )
         y = DW< T >( y.hi(), T( 0 ) );
      T h, l;
      f( x, y, &h, &l );
      set( a, x );
      set( b, y );
      exact( ref, a, b );
      if( !mpfr_zero_p( ref ) )
         worst = std::max( worst, rel_error( ref, h, l ) );
   }
   return worst / U2< T >;
}

static void exact_add( mpfr_ptr r, mpfr_srcptr a, mpfr_srcptr b ) { mpfr_add( r, a, b, MPFR_RNDN ); }
static void exact_mul( mpfr_ptr r, mpfr_srcptr a, mpfr_srcptr b ) { mpfr_mul( r, a, b, MPFR_RNDN ); }
static void exact_div( mpfr_ptr r, mpfr_srcptr a, mpfr_srcptr b ) { mpfr_div( r, a, b, MPFR_RNDN ); }


template< typename T >
static void test_error_free( Source& src )
{
   const int n = SAMPLES;
   bool sum = true, diff = true, fast = true, prod = true, split_ok = true;
   Big a, b, r, s;

   unit::announce( "TwoSum: s + e == a + b exactly, " + std::to_string( n ) + " pairs, half nearly cancelling" );
   for( int i = 0; i < n; ++i ) {
      const T x = src.fp< T >(), y = i % 2 ? src.dw_near_negation( DW< T >( x ) ).hi() : src.fp< T >();
      const rne< T > t = two_sum( x, y );
      mpfr_set_d( a, x, MPFR_RNDN );
      mpfr_add_d( r, a, y, MPFR_RNDN );
      set( s, t.sum, t.error );
      sum &= mpfr_equal_p( r, s ) && t.sum == x + y;
   }
   unit::verdict( sum );

   unit::announce( "TwoDiff: s + e == a - b exactly, " + std::to_string( n ) + " pairs, half nearly cancelling" );
   for( int i = 0; i < n; ++i ) {
      const T x = src.fp< T >(), y = i % 2 ? -src.dw_near_negation( DW< T >( x ) ).hi() : src.fp< T >();
      const rne< T > t = two_diff( x, y );
      mpfr_set_d( a, x, MPFR_RNDN );
      mpfr_sub_d( r, a, y, MPFR_RNDN );
      set( s, t.sum, t.error );
      diff &= mpfr_equal_p( r, s ) && t.sum == x - y;
   }
   unit::verdict( diff );

   unit::announce( "Fast2Sum (|a| >= |b|): s + e == a + b exactly, " + std::to_string( n ) + " pairs" );
   for( int i = 0; i < n; ++i ) {
      T x = src.fp< T >(), y = src.fp< T >();
      if( std::fabs( x ) < std::fabs( y ) )
         std::swap( x, y );
      const rne< T > t = quick_two_sum( x, y );
      mpfr_set_d( a, x, MPFR_RNDN );
      mpfr_add_d( r, a, y, MPFR_RNDN );
      set( s, t.sum, t.error );
      fast &= mpfr_equal_p( r, s );
   }
   unit::verdict( fast );

   unit::announce( "TwoProd (FMA): p + e == a * b exactly, " + std::to_string( n ) + " pairs" );
   for( int i = 0; i < n; ++i ) {
      const T x = src.fp< T >(), y = src.fp< T >();
      const rne< T > t = two_prod( x, y );
      mpfr_set_d( a, x, MPFR_RNDN );
      mpfr_set_d( b, y, MPFR_RNDN );
      mpfr_mul( r, a, b, MPFR_RNDN );
      set( s, t.sum, t.error );
      prod &= mpfr_equal_p( r, s ) && t.sum == x * y;
   }
   unit::verdict( prod );

   unit::announce( "Dekker split: hi + lo == a exactly, " + std::to_string( n ) + " values" );
   for( int i = 0; i < n; ++i ) {
      const T x = src.fp< T >();
      const auto t = split< T, T >( x );
      mpfr_set_d( a, x, MPFR_RNDN );
      set( s, t.sum, t.error );
      split_ok &= mpfr_equal_p( a, s );
   }
   unit::verdict( split_ok );
}

template< typename T >
static void test_dw_algorithms( Source& src )
{
   const std::string n = std::to_string( SAMPLES );
   const double u = std::sqrt( U2< T > );  // 56u^3 in units of u^2 is 56u

   unit::announce( "MaddDWPlusDW, " + n + " pairs, half nearly cancelling" );
   unit::verdict_bound( worst_binary< T >( src, Pairs::Cancelling, []( DW< T > x, DW< T > y, T* h, T* l ) { MaddDWPlusDW( x.hi(), x.lo(), y.hi(), y.lo(), h, l ); }, exact_add ), 2 );

   unit::announce( "AccurateDWPlusDW, " + n + " pairs, half nearly cancelling" );
   unit::verdict_bound( worst_binary< T >( src, Pairs::Cancelling, []( DW< T > x, DW< T > y, T* h, T* l ) { AccurateDWPlusDW( x.hi(), x.lo(), y.hi(), y.lo(), h, l ); }, exact_add ), 3 );

   unit::announce( "SloppyDWPlusDW, " + n + " same-sign pairs (no cancellation)" );
   unit::info( worst_binary< T >( src, Pairs::Positive, []( DW< T > x, DW< T > y, T* h, T* l ) { SloppyDWPlusDW( x.hi(), x.lo(), y.hi(), y.lo(), h, l ); }, exact_add ) );

   unit::announce( "DWPlusFP, " + n + " pairs, half nearly cancelling" );
   unit::verdict_bound( worst_binary< T >( src, Pairs::Cancelling, []( DW< T > x, DW< T > y, T* h, T* l ) { DWPlusFP( x.hi(), x.lo(), y.hi(), h, l ); }, exact_add, Second::FP ), 3 );

   unit::announce( "DWTimesDW2, " + n + " pairs" );
   unit::verdict_bound( worst_binary< T >( src, Pairs::Random, []( DW< T > x, DW< T > y, T* h, T* l ) { DWTimesDW2( x.hi(), x.lo(), y.hi(), y.lo(), h, l ); }, exact_mul ), 5 );

   unit::announce( "DWTimesDW2Unnorm, value of hi + lo, " + n + " pairs" );
   unit::verdict_bound( worst_binary< T >( src, Pairs::Random, []( DW< T > x, DW< T > y, T* h, T* l ) { DWTimesDW2Unnorm( x.hi(), x.lo(), y.hi(), y.lo(), h, l ); }, exact_mul ), 5 );

   unit::announce( "DWTimesDW3, " + n + " pairs" );
   unit::verdict_bound( worst_binary< T >( src, Pairs::Random, []( DW< T > x, DW< T > y, T* h, T* l ) { DWTimesDW3( x.hi(), x.lo(), y.hi(), y.lo(), h, l ); }, exact_mul ), 5 );

   unit::announce( "DWTimesFP1, " + n + " pairs" );
   unit::verdict_bound( worst_binary< T >( src, Pairs::Random, []( DW< T > x, DW< T > y, T* h, T* l ) { DWTimesFP1( x.hi(), x.lo(), y.hi(), h, l ); }, exact_mul, Second::FP ), 2 );

   unit::announce( "DWTimesFP3, " + n + " pairs" );
   unit::verdict_bound( worst_binary< T >( src, Pairs::Random, []( DW< T > x, DW< T > y, T* h, T* l ) { DWTimesFP3( x.hi(), x.lo(), y.hi(), h, l ); }, exact_mul, Second::FP ), 2 );

   unit::announce( "DWDivDW2, " + n + " pairs" );
   unit::verdict_bound( worst_binary< T >( src, Pairs::Random, []( DW< T > x, DW< T > y, T* h, T* l ) { DWDivDW2( x.hi(), x.lo(), y.hi(), y.lo(), h, l ); }, exact_div ), 15 + 56 * u );

   unit::announce( "DWDivDW3, " + n + " pairs" );
   unit::verdict_bound( worst_binary< T >( src, Pairs::Random, []( DW< T > x, DW< T > y, T* h, T* l ) { DWDivDW3( x.hi(), x.lo(), y.hi(), y.lo(), h, l ); }, exact_div ), 9.8 );

   unit::announce( "DWRecip, 1 / y, " + n + " values" );
   unit::info( worst_binary< T >( src, Pairs::Random, []( DW< T >, DW< T > y, T* h, T* l ) { DWRecip( y.hi(), y.lo(), h, l ); },
                                  []( mpfr_ptr r, mpfr_srcptr, mpfr_srcptr b ) { mpfr_ui_div( r, 1, b, MPFR_RNDN ); } ) );

   unit::announce( "DWSqrt, " + n + " positive values" );
   unit::verdict_bound( worst_binary< T >( src, Pairs::Positive, []( DW< T > x, DW< T >, T* h, T* l ) { DWSqrt( x.hi(), x.lo(), h, l ); },
                                            []( mpfr_ptr r, mpfr_srcptr a, mpfr_srcptr ) { mpfr_sqrt( r, a, MPFR_RNDN ); } ), 25.0 / 8 );

   unit::announce( "DWPow2, x^2, " + n + " values" );
   unit::info( worst_binary< T >( src, Pairs::Random, []( DW< T > x, DW< T >, T* h, T* l ) { DWPow2( x.hi(), x.lo(), h, l ); },
                                  []( mpfr_ptr r, mpfr_srcptr a, mpfr_srcptr ) { mpfr_sqr( r, a, MPFR_RNDN ); } ) );

   unit::announce( "DWPow2Unnorm, value of hi + lo, x^2, " + n + " values" );
   unit::info( worst_binary< T >( src, Pairs::Random, []( DW< T > x, DW< T >, T* h, T* l ) { DWPow2Unnorm( x.hi(), x.lo(), h, l ); },
                                  []( mpfr_ptr r, mpfr_srcptr a, mpfr_srcptr ) { mpfr_sqr( r, a, MPFR_RNDN ); } ) );
}

template< typename T, AddMode Add, NormMode Norm >
static void test_pow_add( Source& src, const char* name )
{
   unit::announce( std::string( "DWPowAdd<" ) + name + ">, x^2 + y^2, " + std::to_string( SAMPLES ) + " pairs" );
   unit::info( worst_binary< T >( src, Pairs::Random, []( DW< T > x, DW< T > y, T* h, T* l ) { DWPowAdd< T, Add, Norm >( x.hi(), x.lo(), y.hi(), y.lo(), h, l ); },
                                  []( mpfr_ptr r, mpfr_srcptr a, mpfr_srcptr b ) {
                                     Big t;
                                     mpfr_sqr( r, a, MPFR_RNDN );
                                     mpfr_sqr( t, b, MPFR_RNDN );
                                     mpfr_add( r, r, t, MPFR_RNDN );
                                  } ) );
}

// a*b + c*d with c*d nearly -a*b on every other sample; error normalized per sample by
// K = (|ab| + |cd|) / |ab + cd|.
template< typename T, typename F >
static void test_mul_add( Source& src, const char* name, double bound, F f )
{
   unit::announce( std::string( name ) + ", a*b + c*d, " + std::to_string( SAMPLES ) + " samples, error / K" );
   Big a, b, c, d, ab, cd, ref, mag;
   double worst = 0;
   for( int i = 0; i < SAMPLES; ++i ) {
      const DW< T > x = src.dw< T >(), y = src.dw< T >(), z = src.dw< T >();
      DW< T > w = src.dw< T >();
      if( i % 2 ) {
         set( a, x );
         set( b, y );
         set( c, z );
         mpfr_mul( ab, a, b, MPFR_RNDN );
         src.near_negation( cd, ab );
         mpfr_div( d, cd, c, MPFR_RNDN );
         w = to_dw< T >( d );
      }
      T h, l;
      f( x, y, z, w, &h, &l );
      set( a, x );
      set( b, y );
      set( c, z );
      set( d, w );
      mpfr_mul( ab, a, b, MPFR_RNDN );
      mpfr_mul( cd, c, d, MPFR_RNDN );
      mpfr_add( ref, ab, cd, MPFR_RNDN );
      if( mpfr_zero_p( ref ) )
         continue;
      mpfr_abs( ab, ab, MPFR_RNDN );
      mpfr_abs( cd, cd, MPFR_RNDN );
      mpfr_add( mag, ab, cd, MPFR_RNDN );
      mpfr_div( mag, mag, ref, MPFR_RNDN );
      const double K = std::fabs( mpfr_get_d( mag, MPFR_RNDN ) );
      worst = std::max( worst, rel_error( ref, h, l ) / K );
   }
   unit::verdict_bound( worst / U2< T >, bound, "K u^2" );
}

#define MUL_ADD( FN ) []( DW< T > x, DW< T > y, DW< T > z, DW< T > w, T* h, T* l ) { FN( x.hi(), x.lo(), y.hi(), y.lo(), z.hi(), z.lo(), w.hi(), w.lo(), h, l ); }

template< typename T >
static void run( const char* type, unsigned long seed )
{
   Source src( seed );
   unit::section( std::string( "Error-free transforms, T = " ) + type );
   test_error_free< T >( src );

   unit::section( std::string( "Double-word algorithms vs MPFR (relative error / u^2), T = " ) + type );
   test_dw_algorithms< T >( src );
   test_pow_add< T, AddMode::Madd, NormMode::Normalized >( src, "Madd, Normalized" );
   test_pow_add< T, AddMode::Madd, NormMode::Unnormalized >( src, "Madd, Unnormalized" );
   test_pow_add< T, AddMode::Accurate, NormMode::Normalized >( src, "Accurate, Normalized" );
   test_pow_add< T, AddMode::Accurate, NormMode::Unnormalized >( src, "Accurate, Unnormalized" );
   test_pow_add< T, AddMode::Sloppy, NormMode::Normalized >( src, "Sloppy, Normalized" );
   test_pow_add< T, AddMode::Sloppy, NormMode::Unnormalized >( src, "Sloppy, Unnormalized" );

   unit::section( std::string( "Mul-adds a*b + c*d vs MPFR (relative error / (K u^2)), T = " ) + type );
   test_mul_add< T >( src, "DWMulAdd_Madd_N", 7, MUL_ADD( DWMulAdd_Madd_N ) );
   test_mul_add< T >( src, "DWMulAdd_Accu_N", 8, MUL_ADD( DWMulAdd_Accu_N ) );
   test_mul_add< T >( src, "DWMulAdd_Slop_N", 8, MUL_ADD( DWMulAdd_Slop_N ) );
   test_mul_add< T >( src, "DWMulAdd_Madd_U", 8, MUL_ADD( DWMulAdd_Madd_U ) );
   test_mul_add< T >( src, "DWMulAdd_Accu_U", 10, MUL_ADD( DWMulAdd_Accu_U ) );
   test_mul_add< T >( src, "DWMulAdd_Slop_U", 12, MUL_ADD( DWMulAdd_Slop_U ) );
}

int main()
{
   mpfr_set_default_prec( MPFR_BITS );
   run< double >( "double", 101 );
   run< float >( "float", 102 );
   mpfr_free_cache();
   return unit::finish( "test_arith" );
}
