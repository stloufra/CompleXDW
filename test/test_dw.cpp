// MPFR checks of DW<T> (double and float) and of its operations with XDW<T>.
// Bounds: + - MaddDWPlusDW 2u^2, * DWTimesDW2 5u^2, / DWDivDW2 15u^2 + 56u^3, norm 7u^2, sqrt 25/8 u^2; u = 2^-p.

#include <algorithm>
#include <cfenv>
#include <cmath>
#include <complex>
#include <iostream>
#include <limits>
#include <random>
#include <string>

#include <gmp.h>
#include <mpfr.h>

#include "../XDW.h"
#include "src/test_func.h"

constexpr int SAMPLES = 200000;

static int failures = 0;

static void check( bool condition, const std::string& what )
{
    if( condition )
        return;
    std::cerr << "FAILED: " << what << '\n';
    ++failures;
}

template< typename T >
constexpr double U2 = 1.0 / ( double( 1ull << std::numeric_limits< T >::digits ) * double( 1ull << std::numeric_limits< T >::digits ) );

template< typename T >
static T get( mpfr_t x )
{
    if constexpr( std::is_same_v< T, float > )
        return mpfr_get_flt( x, MPFR_RNDN );
    else
        return mpfr_get_d( x, MPFR_RNDN );
}

// Round to hi, then round the remainder to lo: a normalized DW.
template< typename T >
static DW< T > to_dw( mpfr_t x )
{
    mpfr_t rest;
    mpfr_init2( rest, MPFR_PREC );
    const T hi = get< T >( x );
    mpfr_sub_d( rest, x, hi, MPFR_RNDN );
    const T lo = get< T >( rest );
    mpfr_clear( rest );
    return DW< T >( hi, lo );
}

template< typename T >
static void to_mpfr( mpfr_t out, const DW< T >& x )
{
    mpfr_set_d( out, x.hi(), MPFR_RNDN );
    mpfr_add_d( out, out, x.lo(), MPFR_RNDN );
}

template< typename T >
static double rel_error( mpfr_t exact, const DW< T >& x )
{
    mpfr_t err;
    mpfr_init2( err, MPFR_PREC );
    to_mpfr( err, x );
    mpfr_sub( err, err, exact, MPFR_RNDN );
    mpfr_div( err, err, exact, MPFR_RNDN );
    const double e = std::fabs( mpfr_get_d( err, MPFR_RNDN ) );
    mpfr_clear( err );
    return e;
}

// Random value with full MPFR_PREC significand, random sign and exponent in [-20, 20], so float
// products and quotients of two stay normal.
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

    void draw( mpfr_t x )
    {
        mpfr_urandom( x, state, MPFR_RNDN );
        mpfr_add_d( x, x, 0.5, MPFR_RNDN );
        mpfr_mul_2si( x, x, exponent( rng ), MPFR_RNDN );
        if( rng() & 1 )
            mpfr_neg( x, x, MPFR_RNDN );
    }

    // -x * (1 + 2^-k r): adding it to x cancels about k bits.
    void near_negation( mpfr_t out, mpfr_t x )
    {
        const int k = std::uniform_int_distribution< int >( 1, 100 )( rng );
        mpfr_urandom( out, state, MPFR_RNDN );
        mpfr_mul_2si( out, out, -k, MPFR_RNDN );
        mpfr_add_ui( out, out, 1, MPFR_RNDN );
        mpfr_mul( out, out, x, MPFR_RNDN );
        mpfr_neg( out, out, MPFR_RNDN );
    }
};

struct Worst
{
    double add = 0, sub = 0, mul = 0, div = 0;
    double xmul = 0, xdiv = 0, xadd = 0, norm = 0, sqrt = 0;
};

template< typename T >
static void test_arithmetic( Source& src, Worst& w )
{
    mpfr_t a, b, exact, re, im;
    mpfr_inits2( MPFR_PREC, a, b, exact, re, im, (mpfr_ptr) nullptr );
    for( int i = 0; i < SAMPLES; ++i ) {
        src.draw( a );
        if( i % 2 )
            src.near_negation( b, a );
        else
            src.draw( b );
        const DW< T > x = to_dw< T >( a ), y = to_dw< T >( b );
        to_mpfr( a, x );
        to_mpfr( b, y );

        mpfr_add( exact, a, b, MPFR_RNDN );
        if( !mpfr_zero_p( exact ) )
            w.add = std::max( w.add, rel_error( exact, x + y ) );
        mpfr_sub( exact, a, b, MPFR_RNDN );
        if( !mpfr_zero_p( exact ) )
            w.sub = std::max( w.sub, rel_error( exact, x - y ) );
        // x - (-y') cancels as heavily as x + y'.
        mpfr_neg( b, b, MPFR_RNDN );
        mpfr_sub( exact, a, b, MPFR_RNDN );
        if( !mpfr_zero_p( exact ) )
            w.sub = std::max( w.sub, rel_error( exact, x - ( -y ) ) );
        mpfr_neg( b, b, MPFR_RNDN );
        mpfr_mul( exact, a, b, MPFR_RNDN );
        w.mul = std::max( w.mul, rel_error( exact, x * y ) );
        mpfr_div( exact, a, b, MPFR_RNDN );
        w.div = std::max( w.div, rel_error( exact, x / y ) );

        // XDW z = x + i y with the real DW y, which half the time nearly cancels x.
        const XDW< T > z( x, y );
        check( real( z ) == x && imag( z ) == y, "real/imag return the DW parts" );
        mpfr_mul( re, a, b, MPFR_RNDN );
        mpfr_mul( im, b, b, MPFR_RNDN );
        const XDW< T > zy = z * y;
        w.xmul = std::max( { w.xmul, rel_error( re, real( zy ) ), rel_error( im, imag( zy ) ) } );
        check( y * z == zy, "DW * XDW == XDW * DW" );
        const XDW< T > zq = z / y;
        mpfr_div( re, a, b, MPFR_RNDN );
        mpfr_set_ui( im, 1, MPFR_RNDN );
        w.xdiv = std::max( { w.xdiv, rel_error( re, real( zq ) ), rel_error( im, imag( zq ) ) } );
        const XDW< T > zs = z + y;
        mpfr_add( re, a, b, MPFR_RNDN );
        if( !mpfr_zero_p( re ) )
            w.xadd = std::max( w.xadd, rel_error( re, real( zs ) ) );
        check( imag( zs ) == y && y + z == zs, "XDW + DW keeps imag, commutes" );
        check( real( z - y ) == x - y && imag( z - y ) == y, "XDW - DW" );
        check( real( y - z ) == y - x && imag( y - z ) == -y, "DW - XDW" );
        mpfr_mul( re, a, a, MPFR_RNDN );
        mpfr_mul( im, b, b, MPFR_RNDN );
        mpfr_add( exact, re, im, MPFR_RNDN );
        w.norm = std::max( w.norm, rel_error( exact, norm( z ) ) );

        // sqrt, abs, signbit, min/max and select are exact apart from sqrt's rounding.
        const DW< T > ax = abs( x );
        to_mpfr( re, ax );
        check( mpfr_cmpabs( re, a ) == 0 && mpfr_sgn( re ) >= 0, "abs" );
        check( ( signbit( x ) != 0 ) == ( mpfr_sgn( a ) < 0 ), "signbit" );
        mpfr_abs( exact, a, MPFR_RNDN );
        mpfr_sqrt( exact, exact, MPFR_RNDN );
        w.sqrt = std::max( w.sqrt, rel_error( exact, sqrt( ax ) ) );
        const bool x_smaller = mpfr_cmp( a, b ) < 0;
        check( min( x, y ) == ( x_smaller ? x : y ) && max( x, y ) == ( x_smaller ? y : x ), "min/max" );
        check( select( true, x, y ) == x && select( false, x, y ) == y && select( x_smaller, z, conj( z ) ) == ( x_smaller ? z : conj( z ) ),
               "select" );
        check( max( x, 0. ) == max( x, DW< T >( 0. ) ) && min( 1, x ) == min( DW< T >( 1 ), x ), "min/max with a plain number" );
    }
    mpfr_clears( a, b, exact, re, im, (mpfr_ptr) nullptr );
}

// Zeros must not raise floating-point exceptions (MadGraph checks for them).
template< typename T >
static void test_special()
{
    const DW< T > zero( T( 0 ), T( 0 ) ), negzero( -T( 0 ), -T( 0 ) );
    std::feclearexcept( FE_ALL_EXCEPT );
    const DW< T > r0 = sqrt( zero ), rn = sqrt( negzero );
    check( !std::fetestexcept( FE_INVALID | FE_DIVBYZERO ), "sqrt(+-0) raises no exception" );
    check( r0 == zero && rn == zero, "sqrt(+-0) = 0" );
    check( signbit( negzero ) && !signbit( zero ) && !signbit( abs( negzero ) ), "signbit and abs of -0" );
    check( sqrt( DW< T >( 4 ) ) == DW< T >( 2 ), "sqrt(4) = 2 exactly" );
}

template< typename T >
static void test_order( Source& src )
{
    mpfr_t a, b;
    mpfr_inits2( MPFR_PREC, a, b, (mpfr_ptr) nullptr );
    bool same = true;
    for( int i = 0; i < SAMPLES; ++i ) {
        src.draw( a );
        // Every third pair shares hi, so the order is decided by lo.
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
        to_mpfr( a, x );
        to_mpfr( b, y );
        const int c = mpfr_cmp( a, b );
        same &= ( x < y ) == ( c < 0 ) && ( x > y ) == ( c > 0 ) && ( x == y ) == ( c == 0 )
             && ( x <= y ) == ( c <= 0 ) && ( x >= y ) == ( c >= 0 ) && ( x != y ) == ( c != 0 );
    }
    check( same, "DW comparisons agree with MPFR" );
    mpfr_clears( a, b, (mpfr_ptr) nullptr );
}

template< typename T >
static void test_conversions( Source& src )
{
    mpfr_t a, exact;
    mpfr_inits2( MPFR_PREC, a, exact, (mpfr_ptr) nullptr );
    double worst_split = 0;
    bool exact_back = true, mixed = true;
    for( int i = 0; i < SAMPLES; ++i ) {
        src.draw( a );
        const double d = mpfr_get_d( a, MPFR_RNDN );
        const DW< T > x( d );
        mpfr_set_d( exact, d, MPFR_RNDN );
        if constexpr( std::is_same_v< T, float > )
            worst_split = std::max( worst_split, rel_error( exact, x ) );
        else
            exact_back &= x.hi() == d && x.lo() == 0.0;
        exact_back &= static_cast< double >( x ) == double( x.hi() ) + double( x.lo() );

        const DW< T > y = to_dw< T >( a );
        mixed &= ( y + 0.5 ) == ( y + DW< T >( 0.5 ) ) && ( 2 * y ) == ( DW< T >( 2 ) * y ) && ( y / 3 ) == ( y / DW< T >( 3 ) )
              && ( 1.0 - y ) == ( DW< T >( 1.0 ) - y ) && ( ( y < 0 ) == ( y.hi() < 0 ) );
        const XDW< T > z( y, -y );
        mixed &= ( z * 0.5 ) == ( z * DW< T >( 0.5 ) ) && ( 2 * z ) == ( DW< T >( 2 ) * z ) && ( z + 1.0 ) == ( z + DW< T >( 1.0 ) );
        mixed &= XDW< T >( d, -d ) == XDW< T >( DW< T >( d ), DW< T >( -d ) );
        mixed &= XDW< T >( std::complex< double >( d, -d ) ) == XDW< T >( DW< T >( d ), DW< T >( -d ) );
        const std::complex< double > back = static_cast< std::complex< double > >( z );
        mixed &= back.real() == static_cast< double >( y ) && back.imag() == static_cast< double >( -y );
        mixed &= ( y / z ) == ( XDW< T >( y ) / z ) && ( 0.5 / z ) == ( XDW< T >( 0.5 ) / z ) && ( z - 3 ) == ( z - DW< T >( 3 ) );
    }
    check( exact_back, "DW(double) and conversion back to a wider type are exact" );
    check( mixed, "plain numbers mix with DW and XDW as their DW conversion" );
    if constexpr( std::is_same_v< T, float > ) {
        check( worst_split <= U2< float >, "double into DW<float> within u^2" );
        std::cout << "  double -> DW<float> split: worst relative error / u^2 = " << worst_split / U2< float > << '\n';
    }
    check( static_cast< double >( DW< T >( 123456789 ) ) == 123456789.0, "integer into DW is exact up to 2p bits" );
    mpfr_clears( a, exact, (mpfr_ptr) nullptr );
}

// DW<float> -> DW<double> exact, DW<double> -> DW<float> within u_float^2; both normalized.
static void test_precision_conversion( Source& src )
{
    mpfr_t a, exact;
    mpfr_inits2( MPFR_PREC, a, exact, (mpfr_ptr) nullptr );
    double worst = 0;
    bool up_exact = true, normalized = true;
    for( int i = 0; i < SAMPLES; ++i ) {
        src.draw( a );
        const DW< double > xd = to_dw< double >( a );
        const DW< float > xf = to_dw< float >( a );
        to_mpfr( exact, xf );
        const DW< double > up( xf );
        up_exact &= rel_error( exact, up ) == 0.0;
        normalized &= up.hi() + up.lo() == up.hi();
        to_mpfr( exact, xd );
        const DW< float > down( xd );
        worst = std::max( worst, rel_error( exact, down ) );
        normalized &= down.hi() + down.lo() == down.hi();
    }
    check( up_exact, "DW<double>(DW<float>) is exact" );
    check( normalized, "DW<float>(DW<double>) is normalized" );
    check( worst <= U2< float >, "DW<float>(DW<double>) within u^2" );
    std::cout << "  DW<double> -> DW<float>: worst relative error / u^2 = " << worst / U2< float > << '\n';
    mpfr_clears( a, exact, (mpfr_ptr) nullptr );
}

template< typename T >
static void run( const char* name, unsigned long seed )
{
    Source src( seed );
    Worst w;
    test_arithmetic< T >( src, w );
    test_order< T >( src );
    test_conversions< T >( src );
    test_special< T >();

    const double u2 = U2< T >, u3 = u2 * std::sqrt( u2 );
    const std::string t = std::string( "DW<" ) + name + "> ";
    check( w.add <= 2 * u2, t + "+ within 2u^2" );
    check( w.sub <= 2 * u2, t + "- within 2u^2" );
    check( w.mul <= 5 * u2, t + "* within 5u^2" );
    check( w.div <= 15 * u2 + 56 * u3, t + "/ within 15u^2" );
    check( w.xmul <= 5 * u2, t + "XDW * DW within 5u^2" );
    check( w.xdiv <= 15 * u2 + 56 * u3, t + "XDW / DW within 15u^2" );
    check( w.xadd <= 2 * u2, t + "XDW + DW within 2u^2" );
    check( w.norm <= 7 * u2, t + "norm within 7u^2" );
    check( w.sqrt <= 25.0 / 8 * u2, t + "sqrt within 25/8 u^2" );
    std::cout << t << "worst relative error / u^2:  + " << w.add / u2 << "  - " << w.sub / u2 << "  * " << w.mul / u2
              << "  / " << w.div / u2 << "  XDW*DW " << w.xmul / u2 << "  XDW/DW " << w.xdiv / u2
              << "  XDW+DW " << w.xadd / u2 << "  norm " << w.norm / u2 << "  sqrt " << w.sqrt / u2 << '\n';
}

int main()
{
    mpfr_set_default_prec( MPFR_PREC );
    run< double >( "double", 42 );
    run< float >( "float", 43 );
    Source src( 44 );
    test_precision_conversion( src );
    mpfr_free_cache();
    if( failures )
        std::cerr << failures << " check(s) failed\n";
    return failures == 0 ? 0 : 1;
}
