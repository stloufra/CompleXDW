// DW<T> (double and float) against MPFR: arithmetic within its bounds, the exact functions exact,
// conversions as documented. The XDW side is in test_xdw.cpp.

#include <algorithm>
#include <cfenv>
#include <cmath>
#include <string>

#include "mpfr_support.h"
#include "unit_test.h"

// Worst relative error / u^2 of op(x, y) against exact(x, y); partner(x, i) makes y.
template< typename T, typename Op, typename Exact, typename Partner >
static double worst_op( Source& src, Op op, Exact exact, Partner partner )
{
   Big a, b, ref;
   double worst = 0;
   for( int i = 0; i < SAMPLES; ++i ) {
      const DW< T > x = src.dw< T >(), y = partner( x, i );
      set( a, x );
      set( b, y );
      exact( ref, a, b );
      if( !mpfr_zero_p( ref ) )
         worst = std::max( worst, rel_error( ref, op( x, y ) ) );
   }
   return worst / U2< T >;
}

template< typename T >
static void test_arithmetic( Source& src )
{
   const std::string n = std::to_string( SAMPLES );
   auto random = [&]( const DW< T >&, int ) { return src.dw< T >(); };
   auto cancels_sum = [&]( const DW< T >& x, int i ) { return i % 2 ? src.dw_near_negation( x ) : src.dw< T >(); };
   auto cancels_difference = [&]( const DW< T >& x, int i ) { return i % 2 ? -src.dw_near_negation( x ) : src.dw< T >(); };

   unit::announce( "x + y (MaddDWPlusDW), " + n + " pairs, half nearly cancelling" );
   unit::verdict_bound( worst_op< T >( src, []( DW< T > x, DW< T > y ) { return x + y; },
                                       []( mpfr_ptr r, mpfr_srcptr a, mpfr_srcptr b ) { mpfr_add( r, a, b, MPFR_RNDN ); }, cancels_sum ), 2 );

   unit::announce( "x - y (MaddDWPlusDW), " + n + " pairs, half nearly cancelling" );
   unit::verdict_bound( worst_op< T >( src, []( DW< T > x, DW< T > y ) { return x - y; },
                                       []( mpfr_ptr r, mpfr_srcptr a, mpfr_srcptr b ) { mpfr_sub( r, a, b, MPFR_RNDN ); }, cancels_difference ), 2 );

   unit::announce( "x * y (DWTimesDW2), " + n + " pairs" );
   unit::verdict_bound( worst_op< T >( src, []( DW< T > x, DW< T > y ) { return x * y; },
                                       []( mpfr_ptr r, mpfr_srcptr a, mpfr_srcptr b ) { mpfr_mul( r, a, b, MPFR_RNDN ); }, random ), 5 );

   unit::announce( "x / y (DWDivDW2), " + n + " pairs" );
   unit::verdict_bound( worst_op< T >( src, []( DW< T > x, DW< T > y ) { return x / y; },
                                       []( mpfr_ptr r, mpfr_srcptr a, mpfr_srcptr b ) { mpfr_div( r, a, b, MPFR_RNDN ); }, random ),
                        15 + 56 * std::sqrt( U2< T > ) );

   unit::announce( "sqrt(|x|) (DWSqrt), " + n + " values" );
   unit::verdict_bound( worst_op< T >( src, []( DW< T > x, DW< T > ) { return sqrt( abs( x ) ); },
                                       []( mpfr_ptr r, mpfr_srcptr a, mpfr_srcptr ) { mpfr_abs( r, a, MPFR_RNDN ); mpfr_sqrt( r, r, MPFR_RNDN ); }, random ),
                        25.0 / 8 );
}

template< typename T >
static void test_exact_functions( Source& src )
{
   const std::string n = std::to_string( SAMPLES );
   Big a, b, r;

   unit::announce( "abs(x) == |x| and signbit(x) == (x < 0), " + n + " values" );
   bool abs_ok = true;
   for( int i = 0; i < SAMPLES; ++i ) {
      const DW< T > x = src.dw< T >();
      set( a, x );
      set( r, abs( x ) );
      abs_ok &= mpfr_cmpabs( r, a ) == 0 && mpfr_sgn( r.v ) >= 0 && ( signbit( x ) != 0 ) == ( mpfr_sgn( a.v ) < 0 );
   }
   unit::verdict( abs_ok );

   unit::announce( "min/max return the smaller/larger operand, select(mask, a, b) picks a or b, " + n + " pairs" );
   bool pick_ok = true;
   for( int i = 0; i < SAMPLES; ++i ) {
      const DW< T > x = src.dw< T >(), y = src.dw< T >();
      set( a, x );
      set( b, y );
      const bool x_smaller = mpfr_cmp( a, b ) < 0;
      pick_ok &= min( x, y ) == ( x_smaller ? x : y ) && max( x, y ) == ( x_smaller ? y : x )
              && select( x_smaller, x, y ) == ( x_smaller ? x : y ) && select( !x_smaller, x, y ) == ( x_smaller ? y : x );
   }
   unit::verdict( pick_ok );

   unit::announce( "< <= > >= == != agree with MPFR, " + n + " pairs, a third sharing hi" );
   bool order_ok = true;
   for( int i = 0; i < SAMPLES; ++i ) {
      src.draw( a );
      if( i % 3 == 0 ) {
         constexpr int shift = -2 * std::numeric_limits< T >::digits - 30;
         const T hi = to_dw< T >( a ).hi();
         src.draw( a );
         mpfr_mul_2si( a, a, shift, MPFR_RNDN );
         mpfr_add_d( a, a, hi, MPFR_RNDN );
         src.draw( b );
         mpfr_mul_2si( b, b, shift, MPFR_RNDN );
         mpfr_add_d( b, b, hi, MPFR_RNDN );
      }
      else
         src.draw( b );
      const DW< T > x = to_dw< T >( a ), y = to_dw< T >( b );
      set( a, x );
      set( b, y );
      const int c = mpfr_cmp( a, b );
      order_ok &= ( x < y ) == ( c < 0 ) && ( x > y ) == ( c > 0 ) && ( x == y ) == ( c == 0 )
               && ( x <= y ) == ( c <= 0 ) && ( x >= y ) == ( c >= 0 ) && ( x != y ) == ( c != 0 );
   }
   unit::verdict( order_ok );

   unit::announce( "sqrt(+0) = sqrt(-0) = 0 without floating-point exceptions, sqrt(4) = 2" );
   const DW< T > zero( T( 0 ), T( 0 ) ), negzero( -T( 0 ), -T( 0 ) );
   std::feclearexcept( FE_ALL_EXCEPT );
   const DW< T > r0 = sqrt( zero ), rn = sqrt( negzero );
   const bool quiet = !std::fetestexcept( FE_INVALID | FE_DIVBYZERO );
   unit::verdict( quiet && r0 == zero && rn == zero && sqrt( DW< T >( 4 ) ) == DW< T >( 2 ) );

   unit::announce( "signbit(-0) is set, abs(-0) is +0" );
   unit::verdict( signbit( negzero ) && !signbit( zero ) && !signbit( abs( negzero ) ) );
}

template< typename T >
static void test_conversions( Source& src )
{
   const std::string n = std::to_string( SAMPLES );
   Big a, ref;

   if constexpr( std::is_same_v< T, float > ) {
      unit::announce( "DW<float>(double) splits into hi + lo, " + n + " values" );
      double worst = 0;
      for( int i = 0; i < SAMPLES; ++i ) {
         src.draw( a );
         const double d = mpfr_get_d( a, MPFR_RNDN );
         mpfr_set_d( ref, d, MPFR_RNDN );
         worst = std::max( worst, rel_error( ref, DW< float >( d ) ) );
      }
      unit::verdict_bound( worst / U2< float >, 1 );
   }
   else {
      unit::announce( "DW<double>(double) is (d, 0), " + n + " values" );
      bool ok = true;
      for( int i = 0; i < SAMPLES; ++i ) {
         src.draw( a );
         const double d = mpfr_get_d( a, MPFR_RNDN );
         ok &= DW< double >( d ).hi() == d && DW< double >( d ).lo() == 0.0;
      }
      unit::verdict( ok );
   }

   unit::announce( "static_cast<double>(DW<T>): hi + lo exactly for float, hi for double, " + n + " values" );
   bool back = true;
   for( int i = 0; i < SAMPLES; ++i ) {
      const DW< T > x = src.dw< T >();
      back &= static_cast< double >( x ) == ( std::is_same_v< T, float > ? double( x.hi() ) + double( x.lo() ) : double( x.hi() ) );
   }
   unit::verdict( back );

   unit::announce( "an integer converts exactly: DW<T>(123456789)" );
   unit::verdict( static_cast< double >( DW< T >( 123456789 ) ) == 123456789.0 );

   unit::announce( "plain numbers mix as their DW conversion: x + 0.5, 2 * x, x / 3, 1.0 - x, min/max(x, number), " + n + " values" );
   bool mixed = true;
   for( int i = 0; i < SAMPLES; ++i ) {
      const DW< T > x = src.dw< T >();
      mixed &= ( x + 0.5 ) == ( x + DW< T >( 0.5 ) ) && ( 2 * x ) == ( DW< T >( 2 ) * x ) && ( x / 3 ) == ( x / DW< T >( 3 ) )
            && ( 1.0 - x ) == ( DW< T >( 1.0 ) - x ) && ( ( x < 0 ) == ( x.hi() < 0 ) )
            && max( x, 0. ) == max( x, DW< T >( 0. ) ) && min( 1, x ) == min( DW< T >( 1 ), x );
   }
   unit::verdict( mixed );
}

static void test_precision_conversion( Source& src )
{
   const std::string n = std::to_string( SAMPLES );
   Big a, ref;
   double worst = 0;
   bool up_exact = true, normalized = true;

   unit::announce( "DW<double>(DW<float>) is exact and normalized, " + n + " values" );
   for( int i = 0; i < SAMPLES; ++i ) {
      const DW< float > xf = src.dw< float >();
      const DW< double > up( xf );
      set( ref, xf );
      up_exact &= rel_error( ref, up ) == 0.0 && up.hi() + up.lo() == up.hi();
   }
   unit::verdict( up_exact );

   unit::announce( "DW<float>(DW<double>) rounds to the nearest pair and is normalized, " + n + " values" );
   for( int i = 0; i < SAMPLES; ++i ) {
      const DW< double > xd = src.dw< double >();
      const DW< float > down( xd );
      set( ref, xd );
      worst = std::max( worst, rel_error( ref, down ) );
      normalized &= down.hi() + down.lo() == down.hi();
   }
   unit::verdict_bound( normalized ? worst / U2< float > : 1e9, 1 );
}

template< typename T >
static void run( const char* type, unsigned long seed )
{
   Source src( seed );
   unit::section( std::string( "DW<" ) + type + ">: arithmetic vs MPFR, relative error in units of u^2" );
   test_arithmetic< T >( src );
   unit::section( std::string( "DW<" ) + type + ">: abs, signbit, min, max, select, comparisons, zeros" );
   test_exact_functions< T >( src );
   unit::section( std::string( "DW<" ) + type + ">: conversions" );
   test_conversions< T >( src );
}

int main()
{
   mpfr_set_default_prec( MPFR_BITS );
   run< double >( "double", 42 );
   run< float >( "float", 43 );
   Source src( 44 );
   unit::section( "DW<double> <-> DW<float>" );
   test_precision_conversion( src );
   mpfr_free_cache();
   return unit::finish( "test_dw" );
}
