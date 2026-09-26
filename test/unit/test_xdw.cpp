// XDW<T> (double and float) against MPFR: every mul/div/norm mode within its bound, the XDW-DW
// operations, construction, conversions and operators.

#include <algorithm>
#include <cmath>
#include <complex>
#include <string>
#include <vector>

#include "mpfr_support.h"
#include "unit_test.h"

using namespace XDW_ARTH;

constexpr int SAMPLES_CX = 50000;
// Division is checked on numerators a*c + b*d, b*c - a*d with conditioning K <= this: the bound
// below assumes a well-conditioned numerator, K itself is what the conditioning studies measure.
constexpr double DIV_COND_LIMIT = 10.0;
constexpr double DIV_BOUND = 29.0;

template< typename T >
static XDW< T > random_xdw( Source& src )
{
   return XDW< T >( src.dw< T >(), src.dw< T >() );
}

template< typename T >
static double cx_error( mpfr_srcptr re, mpfr_srcptr im, const XDW< T >& z )
{
   return std::max( rel_error( re, real( z ) ), rel_error( im, imag( z ) ) );
}

// |p| + |q| over |p + q|.
static double conditioning( mpfr_srcptr p, mpfr_srcptr q )
{
   Big s, m, t;
   mpfr_add( s, p, q, MPFR_RNDN );
   mpfr_abs( m, p, MPFR_RNDN );
   mpfr_abs( t, q, MPFR_RNDN );
   mpfr_add( m, m, t, MPFR_RNDN );
   mpfr_div( m, m, s, MPFR_RNDN );
   return std::fabs( mpfr_get_d( m, MPFR_RNDN ) );
}

template< typename T >
static void test_construction( Source& src )
{
   const std::string n = std::to_string( SAMPLES_CX );

   unit::announce( "XDW(DW re, DW im), real, imag, conj and parts round-trip, " + n + " values" );
   bool parts = true;
   for( int i = 0; i < SAMPLES_CX; ++i ) {
      const DW< T > x = src.dw< T >(), y = src.dw< T >();
      const XDW< T > z( x, y );
      const XDW< T > c = conj( z );
      parts &= real( z ) == x && imag( z ) == y && z == XDW< T >( x.hi(), x.lo(), y.hi(), y.lo() )
            && real( c ) == x && imag( c ) == -y && XDW< T >( x ) == XDW< T >( x, DW< T >( 0 ) );
   }
   unit::verdict( parts );

   unit::announce( "plain numbers and std::complex<double> are split like DW, " + n + " values" );
   bool split = true;
   for( int i = 0; i < SAMPLES_CX; ++i ) {
      const double re = static_cast< double >( src.dw< double >() ), im = static_cast< double >( src.dw< double >() );
      const XDW< T > expected{ DW< T >( re ), DW< T >( im ) };
      split &= XDW< T >( re, im ) == expected && XDW< T >( std::complex< double >( re, im ) ) == expected;
   }
   unit::verdict( split );

   unit::announce( "static_cast<std::complex<double>>: each part as static_cast<double>(DW), " + n + " values" );
   bool back = true;
   for( int i = 0; i < SAMPLES_CX; ++i ) {
      const XDW< T > z = random_xdw< T >( src );
      const std::complex< double > c = static_cast< std::complex< double > >( z );
      back &= c.real() == static_cast< double >( real( z ) ) && c.imag() == static_cast< double >( imag( z ) );
   }
   unit::verdict( back );
}

template< typename T >
static void test_add_sub( Source& src )
{
   unit::announce( "z + w and z - w (componentwise MaddDWPlusDW), " + std::to_string( SAMPLES_CX ) + " pairs, half nearly cancelling" );
   Big ar, ai, br, bi, re, im;
   double worst = 0;
   for( int i = 0; i < SAMPLES_CX; ++i ) {
      const XDW< T > z = random_xdw< T >( src );
      const XDW< T > w = i % 2 ? XDW< T >( src.dw_near_negation( real( z ) ), src.dw_near_negation( imag( z ) ) ) : random_xdw< T >( src );
      set( ar, real( z ) );
      set( ai, imag( z ) );
      set( br, real( w ) );
      set( bi, imag( w ) );
      mpfr_add( re, ar, br, MPFR_RNDN );
      mpfr_add( im, ai, bi, MPFR_RNDN );
      if( !mpfr_zero_p( re ) && !mpfr_zero_p( im ) )
         worst = std::max( worst, cx_error( re, im, z + w ) );
      mpfr_sub( re, ar, br, MPFR_RNDN );
      mpfr_sub( im, ai, bi, MPFR_RNDN );
      worst = std::max( worst, cx_error( re, im, z - w ) );
   }
   unit::verdict_bound( worst / U2< T >, 2 );
}

// Each component is a mul-add: its error is judged against that component's own conditioning K.
template< typename T, AddMode Add, NormMode Norm >
static void test_mul_mode( Source& src, const char* name, double bound )
{
   unit::announce( std::string( "mul<" ) + name + ">, " + std::to_string( SAMPLES_CX ) + " pairs, half with a nearly cancelling imaginary part" );
   Big ar, ai, br, bi, p, q, re, im;
   double worst = 0;
   for( int i = 0; i < SAMPLES_CX; ++i ) {
      const XDW< T > z = random_xdw< T >( src );
      XDW< T > w = random_xdw< T >( src );
      set( ar, real( z ) );
      set( ai, imag( z ) );
      set( br, real( w ) );
      if( i % 2 ) {
         mpfr_mul( p, ai, br, MPFR_RNDN );
         src.near_negation( q, p );
         mpfr_div( q, q, ar, MPFR_RNDN );
         w = XDW< T >( real( w ), to_dw< T >( q ) );
      }
      set( bi, imag( w ) );
      const XDW< T > r = XDW< T >::template mul< Add, Norm >( z, w );

      mpfr_mul( p, ar, br, MPFR_RNDN );
      mpfr_mul( q, ai, bi, MPFR_RNDN );
      mpfr_neg( q, q, MPFR_RNDN );
      mpfr_add( re, p, q, MPFR_RNDN );
      if( !mpfr_zero_p( re ) )
         worst = std::max( worst, rel_error( re, real( r ) ) / conditioning( p, q ) );
      mpfr_mul( p, ar, bi, MPFR_RNDN );
      mpfr_mul( q, ai, br, MPFR_RNDN );
      mpfr_add( im, p, q, MPFR_RNDN );
      if( !mpfr_zero_p( im ) )
         worst = std::max( worst, rel_error( im, imag( r ) ) / conditioning( p, q ) );
   }
   unit::verdict_bound( worst / U2< T >, bound, "K u^2" );
}

// Inputs with well-conditioned numerators, drawn once and shared by the 12 division modes.
template< typename T >
struct DivCase
{
   XDW< T > z, w;
};

template< typename T >
static std::vector< DivCase< T > > division_cases( Source& src )
{
   std::vector< DivCase< T > > cases;
   Big ar, ai, br, bi, p, q;
   while( static_cast< int >( cases.size() ) < SAMPLES_CX ) {
      const XDW< T > z = random_xdw< T >( src ), w = random_xdw< T >( src );
      set( ar, real( z ) );
      set( ai, imag( z ) );
      set( br, real( w ) );
      set( bi, imag( w ) );
      mpfr_mul( p, ar, br, MPFR_RNDN );
      mpfr_mul( q, ai, bi, MPFR_RNDN );
      const double k_re = conditioning( p, q );
      mpfr_mul( p, ai, br, MPFR_RNDN );
      mpfr_mul( q, ar, bi, MPFR_RNDN );
      mpfr_neg( q, q, MPFR_RNDN );
      if( std::max( k_re, conditioning( p, q ) ) <= DIV_COND_LIMIT )
         cases.push_back( { z, w } );
   }
   return cases;
}

template< typename T, DivMode Div, AddMode Add, NormMode Norm >
static void test_div_mode( const std::vector< DivCase< T > >& cases, const char* name )
{
   unit::announce( std::string( "div<" ) + name + ">, " + std::to_string( cases.size() ) + " pairs with numerator K <= 10" );
   Big ar, ai, br, bi, p, q, d, re, im;
   double worst = 0;
   for( const DivCase< T >& c : cases ) {
      set( ar, real( c.z ) );
      set( ai, imag( c.z ) );
      set( br, real( c.w ) );
      set( bi, imag( c.w ) );
      mpfr_sqr( d, br, MPFR_RNDN );
      mpfr_sqr( p, bi, MPFR_RNDN );
      mpfr_add( d, d, p, MPFR_RNDN );
      mpfr_mul( p, ar, br, MPFR_RNDN );
      mpfr_mul( q, ai, bi, MPFR_RNDN );
      mpfr_add( re, p, q, MPFR_RNDN );
      mpfr_div( re, re, d, MPFR_RNDN );
      mpfr_mul( p, ai, br, MPFR_RNDN );
      mpfr_mul( q, ar, bi, MPFR_RNDN );
      mpfr_sub( im, p, q, MPFR_RNDN );
      mpfr_div( im, im, d, MPFR_RNDN );
      worst = std::max( worst, cx_error( re, im, XDW< T >::template div< Div, Add, Norm >( c.z, c.w ) ) );
   }
   unit::verdict_bound( worst / U2< T >, DIV_BOUND );
}

template< typename T, AddMode Add, NormMode Norm >
static void test_norm_mode( Source& src, const char* name )
{
   unit::announce( std::string( "norm<" ) + name + ">, |z|^2, " + std::to_string( SAMPLES_CX ) + " values" );
   Big ar, ai, re, t;
   double worst = 0;
   for( int i = 0; i < SAMPLES_CX; ++i ) {
      const XDW< T > z = random_xdw< T >( src );
      set( ar, real( z ) );
      set( ai, imag( z ) );
      mpfr_sqr( re, ar, MPFR_RNDN );
      mpfr_sqr( t, ai, MPFR_RNDN );
      mpfr_add( re, re, t, MPFR_RNDN );
      worst = std::max( worst, rel_error( re, norm< T, Add, Norm >( z ) ) );
   }
   unit::verdict_bound( worst / U2< T >, 7 );
}

// (3 + 4i) / (1 + 2i) = 2.2 - 0.4i: the high words must be T(2.2) and T(-0.4).
template< typename T, DivMode Div, AddMode Add, NormMode Norm >
static bool exact_quotient()
{
   const XDW< T > q = XDW< T >::template div< Div, Add, Norm >( XDW< T >( 3, 4 ), XDW< T >( 1, 2 ) );
   return q.re_h() == T( 2.2 ) && q.im_h() == T( -0.4 );
}

template< typename T, DivMode Div >
static bool exact_quotient_all_add_norm()
{
   return exact_quotient< T, Div, AddMode::Madd, NormMode::Normalized >() && exact_quotient< T, Div, AddMode::Madd, NormMode::Unnormalized >()
       && exact_quotient< T, Div, AddMode::Accurate, NormMode::Normalized >() && exact_quotient< T, Div, AddMode::Accurate, NormMode::Unnormalized >()
       && exact_quotient< T, Div, AddMode::Sloppy, NormMode::Normalized >() && exact_quotient< T, Div, AddMode::Sloppy, NormMode::Unnormalized >();
}

template< typename T >
static void test_modes( Source& src )
{
   test_mul_mode< T, AddMode::Madd, NormMode::Normalized >( src, "Madd, Normalized", 7 );
   test_mul_mode< T, AddMode::Madd, NormMode::Unnormalized >( src, "Madd, Unnormalized", 8 );
   test_mul_mode< T, AddMode::Accurate, NormMode::Normalized >( src, "Accurate, Normalized", 8 );
   test_mul_mode< T, AddMode::Accurate, NormMode::Unnormalized >( src, "Accurate, Unnormalized", 10 );
   test_mul_mode< T, AddMode::Sloppy, NormMode::Normalized >( src, "Sloppy, Normalized", 8 );
   test_mul_mode< T, AddMode::Sloppy, NormMode::Unnormalized >( src, "Sloppy, Unnormalized", 12 );

   const auto cases = division_cases< T >( src );
   test_div_mode< T, DivMode::Div2, AddMode::Madd, NormMode::Normalized >( cases, "Div2, Madd, Normalized" );
   test_div_mode< T, DivMode::Div2, AddMode::Madd, NormMode::Unnormalized >( cases, "Div2, Madd, Unnormalized" );
   test_div_mode< T, DivMode::Div2, AddMode::Accurate, NormMode::Normalized >( cases, "Div2, Accurate, Normalized" );
   test_div_mode< T, DivMode::Div2, AddMode::Accurate, NormMode::Unnormalized >( cases, "Div2, Accurate, Unnormalized" );
   test_div_mode< T, DivMode::Div2, AddMode::Sloppy, NormMode::Normalized >( cases, "Div2, Sloppy, Normalized" );
   test_div_mode< T, DivMode::Div2, AddMode::Sloppy, NormMode::Unnormalized >( cases, "Div2, Sloppy, Unnormalized" );
   test_div_mode< T, DivMode::Div3, AddMode::Madd, NormMode::Normalized >( cases, "Div3, Madd, Normalized" );
   test_div_mode< T, DivMode::Div3, AddMode::Madd, NormMode::Unnormalized >( cases, "Div3, Madd, Unnormalized" );
   test_div_mode< T, DivMode::Div3, AddMode::Accurate, NormMode::Normalized >( cases, "Div3, Accurate, Normalized" );
   test_div_mode< T, DivMode::Div3, AddMode::Accurate, NormMode::Unnormalized >( cases, "Div3, Accurate, Unnormalized" );
   test_div_mode< T, DivMode::Div3, AddMode::Sloppy, NormMode::Normalized >( cases, "Div3, Sloppy, Normalized" );
   test_div_mode< T, DivMode::Div3, AddMode::Sloppy, NormMode::Unnormalized >( cases, "Div3, Sloppy, Unnormalized" );

   unit::announce( "(3 + 4i) / (1 + 2i) = 2.2 - 0.4i in all 12 division modes" );
   unit::verdict( exact_quotient_all_add_norm< T, DivMode::Div2 >() && exact_quotient_all_add_norm< T, DivMode::Div3 >() );

   test_norm_mode< T, AddMode::Madd, NormMode::Normalized >( src, "Madd, Normalized" );
   test_norm_mode< T, AddMode::Madd, NormMode::Unnormalized >( src, "Madd, Unnormalized" );
   test_norm_mode< T, AddMode::Accurate, NormMode::Normalized >( src, "Accurate, Normalized" );
   test_norm_mode< T, AddMode::Accurate, NormMode::Unnormalized >( src, "Accurate, Unnormalized" );
   test_norm_mode< T, AddMode::Sloppy, NormMode::Normalized >( src, "Sloppy, Normalized" );
   test_norm_mode< T, AddMode::Sloppy, NormMode::Unnormalized >( src, "Sloppy, Unnormalized" );
}

// z = x + i y with a real DW operand y, which nearly cancels x on every other sample.
template< typename T >
static void test_with_dw( Source& src )
{
   const std::string n = std::to_string( SAMPLES_CX );
   Big a, b, re, im;
   double w_mul = 0, w_div = 0, w_add = 0;
   bool rest = true;

   unit::announce( "XDW * DW, XDW / DW, XDW + DW vs MPFR, XDW - DW, DW - XDW, DW / XDW, " + n + " pairs, half nearly cancelling" );
   for( int i = 0; i < SAMPLES_CX; ++i ) {
      const DW< T > x = src.dw< T >();
      const DW< T > y = i % 2 ? src.dw_near_negation( x ) : src.dw< T >();
      const XDW< T > z( x, y );
      set( a, x );
      set( b, y );

      mpfr_mul( re, a, b, MPFR_RNDN );
      mpfr_sqr( im, b, MPFR_RNDN );
      w_mul = std::max( w_mul, cx_error( re, im, z * y ) );
      mpfr_div( re, a, b, MPFR_RNDN );
      mpfr_set_ui( im, 1, MPFR_RNDN );
      w_div = std::max( w_div, cx_error( re, im, z / y ) );
      mpfr_add( re, a, b, MPFR_RNDN );
      if( !mpfr_zero_p( re ) )
         w_add = std::max( w_add, rel_error( re, real( z + y ) ) );

      rest &= y * z == z * y && y + z == z + y && imag( z + y ) == y;
      rest &= real( z - y ) == x - y && imag( z - y ) == y && real( y - z ) == y - x && imag( y - z ) == -y;
      rest &= y / z == XDW< T >( y ) / z;
   }
   const bool ok = w_mul <= 5 * U2< T > && w_div <= ( 15 + 56 * std::sqrt( U2< T > ) ) * U2< T > && w_add <= 2 * U2< T > && rest;
   std::ostringstream detail;
   detail << std::setprecision( 3 ) << "worst * " << w_mul / U2< T > << ", / " << w_div / U2< T > << ", + " << w_add / U2< T >
          << " u^2 (bounds 5, 15, 2), the rest exact";
   unit::verdict( ok, detail.str() );
}

template< typename T >
static void test_operators( Source& src )
{
   const std::string n = std::to_string( SAMPLES_CX );

   unit::announce( "a + b, a - b, a * b, a / b equal add, sub, mul<>, div<> with the default modes, " + n + " pairs" );
   bool defaults = true;
   for( int i = 0; i < SAMPLES_CX; ++i ) {
      const XDW< T > a = random_xdw< T >( src ), b = random_xdw< T >( src );
      defaults &= a + b == XDW< T >::add( a, b ) && a - b == XDW< T >::sub( a, b ) && a * b == XDW< T >::mul( a, b )
               && a / b == XDW< T >::div( a, b );
   }
   unit::verdict( defaults );

   unit::announce( "+= -= *= /= equal the binary operators, unary + and -, " + n + " pairs" );
   bool compound = true;
   for( int i = 0; i < SAMPLES_CX; ++i ) {
      const XDW< T > a = random_xdw< T >( src ), b = random_xdw< T >( src );
      XDW< T > c = a, d = a, e = a, f = a;
      c += b;
      d -= b;
      e *= b;
      f /= b;
      compound &= c == a + b && d == a - b && e == a * b && f == a / b && +a == a
               && -a == XDW< T >( -real( a ), -imag( a ) );
   }
   unit::verdict( compound );

   unit::announce( "plain numbers mix as their DW conversion: z * 0.5, 2 * z, z + 1.0, 3 - z, 0.5 / z, " + n + " values" );
   bool mixed = true;
   for( int i = 0; i < SAMPLES_CX; ++i ) {
      const XDW< T > z = random_xdw< T >( src );
      mixed &= z * 0.5 == z * DW< T >( 0.5 ) && 2 * z == DW< T >( 2 ) * z && z + 1.0 == z + DW< T >( 1.0 )
            && 3 - z == DW< T >( 3 ) - z && 0.5 / z == XDW< T >( 0.5 ) / z;
   }
   unit::verdict( mixed );

   unit::announce( "select(mask, z, w) picks z or w, " + n + " pairs" );
   bool picks = true;
   for( int i = 0; i < SAMPLES_CX; ++i ) {
      const XDW< T > z = random_xdw< T >( src ), w = random_xdw< T >( src );
      const bool m = real( z ) < real( w );
      picks &= select( m, z, w ) == ( m ? z : w ) && select( !m, z, w ) == ( m ? w : z );
   }
   unit::verdict( picks );
}

template< typename T >
static void run( const char* type, unsigned long seed )
{
   Source src( seed );
   const std::string t = std::string( "XDW<" ) + type + ">: ";
   unit::section( t + "construction and conversions" );
   test_construction< T >( src );
   unit::section( t + "arithmetic vs MPFR (relative error per component)" );
   test_add_sub< T >( src );
   test_modes< T >( src );
   unit::section( t + "with a real DW operand" );
   test_with_dw< T >( src );
   unit::section( t + "operators" );
   test_operators< T >( src );
}

int main()
{
   mpfr_set_default_prec( MPFR_BITS );
   run< double >( "double", 52 );
   run< float >( "float", 53 );
   mpfr_free_cache();
   return unit::finish( "test_xdw" );
}
