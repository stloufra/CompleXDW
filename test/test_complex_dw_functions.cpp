#include <algorithm>
#include <iostream>
#include <random>
#include <source_location>
#include <stdexcept>
#include <vector>

#include <mpfr.h>

#include "../XDW.h"
#include "../XDWSpan.h"
#include "src/test_func.h"

using namespace XDW_ARTH;

constexpr int SAMPLES = 20000;
constexpr double UNIT_ROUNDOFF_SQUARED = 0x1p-106;
constexpr double NORM_ERROR_BOUND = 7.0 * UNIT_ROUNDOFF_SQUARED;
constexpr double DIV_ERROR_BOUND = 29.0 * UNIT_ROUNDOFF_SQUARED;
// Numerator terms (a*c+b*d, b*c-a*d) more ill-conditioned than this are skipped. DWMulAdd_Madd_N's
// own error bound is K*7u^2, so even a modest numerator condition number K swamps a small flat
// bound like DIV_ERROR_BOUND; this keeps K close to 1, matching "assuming εN is negligible" in
// the CompleXDW paper's error analysis. test_complex_dw_conditioning.cpp studies K itself, for
// multiplication's equivalent ac-bd/ad+bc terms.
constexpr double DIV_COND_LIMIT = 10.0;

static int failures = 0;

static void check( bool condition, const char* what )
{
    if( condition )
        return;
    std::cerr << "FAILED: " << what << '\n';
    ++failures;
}

static bool has_parts( const XDW< double >& z, double reh, double rel, double imh, double iml )
{
    return z.re_h() == reh && z.re_l() == rel && z.im_h() == imh && z.im_l() == iml;
}

static void dw_to_mpfr( mpfr_t out, double high, double low )
{
    mpfr_set_d( out, high, MPFR_RNDN );
    mpfr_add_d( out, out, low, MPFR_RNDN );
}

static void test_conj_real_imag_norm( std::mt19937_64& rng )
{
    double worst_norm_error = 0.0;

    for( int i = 0; i < SAMPLES; ++i ) {
        const auto z = generate_random_dw_single( rng );

        check( has_parts( conj( z ), z.re_h(), z.re_l(), -z.im_h(), -z.im_l() ), "conj" );
        check( real( z ) == DW< double >( z.re_h(), z.re_l() ), "real" );
        check( imag( z ) == DW< double >( z.im_h(), z.im_l() ), "imag" );

        const DW< double > n = norm( z );

        mpfr_t re, im, reference, term;
        mpfr_inits2( MPFR_PREC, re, im, reference, term, (mpfr_ptr) nullptr );
        dw_to_mpfr( re, z.re_h(), z.re_l() );
        dw_to_mpfr( im, z.im_h(), z.im_l() );
        mpfr_mul( reference, re, re, MPFR_RNDN );
        mpfr_mul( term, im, im, MPFR_RNDN );
        mpfr_add( reference, reference, term, MPFR_RNDN );

        worst_norm_error = std::max( worst_norm_error, relative_error( reference, n.hi(), n.lo(), reference, MPFR_RNDN ) );
        mpfr_clears( re, im, reference, term, (mpfr_ptr) nullptr );
    }

    check( worst_norm_error <= NORM_ERROR_BOUND, "norm relative error within 7u^2" );
    std::cout << "norm worst relative error / u^2 = " << worst_norm_error / UNIT_ROUNDOFF_SQUARED << '\n';
}

static void test_div_exact()
{
    // (3+4i)/(1+2i) = 2.2 - 0.4i, exact in double, across a representative set of (Div,Add,Norm).
    const XDW< double > a( 3.0, 4.0 ), b( 1.0, 2.0 );

    auto check_exact = [&]( const XDW< double >& q, const char* what ) {
        check( q.re_h() == 2.2 && q.im_h() == -0.4, what );
    };

    check_exact( a / b, "operator/ default" );
    check_exact( XDW< double >::div( a, b ), "div<> default" );
    check_exact( ( XDW< double >::div< DivMode::Div2, AddMode::Madd, NormMode::Normalized >( a, b ) ), "div<Div2,Madd,Normalized>" );
    check_exact( ( XDW< double >::div< DivMode::Div2, AddMode::Accurate, NormMode::Unnormalized >( a, b ) ), "div<Div2,Accurate,Unnormalized>" );
    check_exact( ( XDW< double >::div< DivMode::Div2, AddMode::Sloppy, NormMode::Unnormalized >( a, b ) ), "div<Div2,Sloppy,Unnormalized>" );
    check_exact( ( XDW< double >::div< DivMode::Div3, AddMode::Madd, NormMode::Normalized >( a, b ) ), "div<Div3,Madd,Normalized>" );
    check_exact( ( XDW< double >::div< DivMode::Div3, AddMode::Accurate, NormMode::Normalized >( a, b ) ), "div<Div3,Accurate,Normalized>" );
    check_exact( ( XDW< double >::div< DivMode::Div3, AddMode::Sloppy, NormMode::Unnormalized >( a, b ) ), "div<Div3,Sloppy,Unnormalized>" );

    auto c = a; c /= b;
    check( c.re_h() == 2.2 && c.im_h() == -0.4, "operator/=" );
}

// a*c+b*d and b*c-a*d individually, relative to the sum of the magnitude of their two terms.
static bool numerator_conditioned( mpfr_t A, mpfr_t B, mpfr_t C, mpfr_t D, mpfr_t num_re, mpfr_t num_im )
{
    mpfr_t t1, t2, cond;
    mpfr_inits2( MPFR_PREC, t1, t2, cond, (mpfr_ptr) nullptr );

    mpfr_mul( t1, A, C, MPFR_RNDN );
    mpfr_mul( t2, B, D, MPFR_RNDN );
    mpfr_add( num_re, t1, t2, MPFR_RNDN );
    mpfr_abs( t1, t1, MPFR_RNDN );
    mpfr_abs( t2, t2, MPFR_RNDN );
    mpfr_add( cond, t1, t2, MPFR_RNDN );
    mpfr_abs( t2, num_re, MPFR_RNDN );
    mpfr_div( cond, cond, t2, MPFR_RNDN );
    bool ok = mpfr_cmp_d( cond, DIV_COND_LIMIT ) <= 0;

    mpfr_mul( t1, B, C, MPFR_RNDN );
    mpfr_mul( t2, A, D, MPFR_RNDN );
    mpfr_sub( num_im, t1, t2, MPFR_RNDN );
    mpfr_abs( t1, t1, MPFR_RNDN );
    mpfr_abs( t2, t2, MPFR_RNDN );
    mpfr_add( cond, t1, t2, MPFR_RNDN );
    mpfr_abs( t2, num_im, MPFR_RNDN );
    mpfr_div( cond, cond, t2, MPFR_RNDN );
    ok = ok && mpfr_cmp_d( cond, DIV_COND_LIMIT ) <= 0;

    mpfr_clears( t1, t2, cond, (mpfr_ptr) nullptr );
    return ok;
}

static void test_div( std::mt19937_64& rng )
{
    test_div_exact();

    double worst_error = 0.0;
    int checked = 0;

    while( checked < SAMPLES ) {
        const auto z1 = generate_random_dw_single( rng );
        const auto z2 = generate_random_dw_single( rng );
        if( z2.re_h() == 0.0 && z2.re_l() == 0.0 && z2.im_h() == 0.0 && z2.im_l() == 0.0 )
            continue;

        mpfr_t A, B, C, D, num_re, num_im, denom, ref_re, ref_im, t1, t2;
        mpfr_inits2( MPFR_PREC, A, B, C, D, num_re, num_im, denom, ref_re, ref_im, t1, t2, (mpfr_ptr) nullptr );
        dw_to_mpfr( A, z1.re_h(), z1.re_l() );
        dw_to_mpfr( B, z1.im_h(), z1.im_l() );
        dw_to_mpfr( C, z2.re_h(), z2.re_l() );
        dw_to_mpfr( D, z2.im_h(), z2.im_l() );

        if( !numerator_conditioned( A, B, C, D, num_re, num_im ) ) {
            mpfr_clears( A, B, C, D, num_re, num_im, denom, ref_re, ref_im, t1, t2, (mpfr_ptr) nullptr );
            continue;
        }
        ++checked;

        mpfr_mul( t1, C, C, MPFR_RNDN );
        mpfr_mul( t2, D, D, MPFR_RNDN );
        mpfr_add( denom, t1, t2, MPFR_RNDN );
        mpfr_div( ref_re, num_re, denom, MPFR_RNDN );
        mpfr_div( ref_im, num_im, denom, MPFR_RNDN );

        auto measure = [&]( const XDW< double >& q ) {
            double err_re = relative_error( ref_re, q.re_h(), q.re_l(), ref_re, MPFR_RNDN );
            double err_im = relative_error( ref_im, q.im_h(), q.im_l(), ref_im, MPFR_RNDN );
            worst_error = std::max( { worst_error, err_re, err_im } );
        };

        measure( z1 / z2 );
        measure( ( XDW< double >::div< DivMode::Div2, AddMode::Madd, NormMode::Unnormalized >( z1, z2 ) ) );
        measure( ( XDW< double >::div< DivMode::Div3, AddMode::Accurate, NormMode::Normalized >( z1, z2 ) ) );
        measure( ( XDW< double >::div< DivMode::Div3, AddMode::Sloppy, NormMode::Unnormalized >( z1, z2 ) ) );

        mpfr_clears( A, B, C, D, num_re, num_im, denom, ref_re, ref_im, t1, t2, (mpfr_ptr) nullptr );
    }

    check( worst_error <= DIV_ERROR_BOUND, "operator/ and div<> relative error within bound" );
    std::cout << "div worst relative error / u^2 = " << worst_error / UNIT_ROUNDOFF_SQUARED << '\n';
}

struct SoA
{
    std::vector< double > re_h, re_l, im_h, im_l;

    explicit SoA( std::size_t n ) : re_h( n ), re_l( n ), im_h( n ), im_l( n ) {}

    XDWSpan< double > span() { return { re_h.data(), re_l.data(), im_h.data(), im_l.data(), re_h.size() }; }
};

static bool identical( const XDW< double >& x, const XDW< double >& y )
{
    return has_parts( x, y.re_h(), y.re_l(), y.im_h(), y.im_l() );
}

// Span mul/div must match the per-element XDW ops bit for bit.
template< AddMode Add, NormMode Norm >
static void check_span_mul( SoA& a, SoA& b )
{
    SoA out( a.re_h.size() );
    mul< Add, Norm >( out.span(), a.span(), b.span() );
    bool same = true;
    for( std::size_t i = 0; i < out.re_h.size(); ++i )
        same &= identical( out.span().load( i ), XDW< double >::mul< Add, Norm >( a.span().load( i ), b.span().load( i ) ) );
    check( same, std::source_location::current().function_name() );
}

template< DivMode Div, AddMode Add, NormMode Norm >
static void check_span_div( SoA& a, SoA& b )
{
    SoA out( a.re_h.size() );
    div< Div, Add, Norm >( out.span(), a.span(), b.span() );
    bool same = true;
    for( std::size_t i = 0; i < out.re_h.size(); ++i )
        same &= identical( out.span().load( i ), XDW< double >::div< Div, Add, Norm >( a.span().load( i ), b.span().load( i ) ) );
    check( same, std::source_location::current().function_name() );
}

template< AddMode Add, NormMode Norm >
static void check_span_modes( SoA& a, SoA& b )
{
    check_span_mul< Add, Norm >( a, b );
    check_span_div< DivMode::Div2, Add, Norm >( a, b );
    check_span_div< DivMode::Div3, Add, Norm >( a, b );
}

static void test_span( std::mt19937_64& rng )
{
    constexpr std::size_t n = 1000;
    SoA a( n ), b( n );
    for( std::size_t i = 0; i < n; ++i ) {
        a.span().store( i, generate_random_dw_single( rng ) );
        b.span().store( i, generate_random_dw_single( rng ) );
    }

    check_span_modes< AddMode::Madd, NormMode::Normalized >( a, b );
    check_span_modes< AddMode::Madd, NormMode::Unnormalized >( a, b );
    check_span_modes< AddMode::Accurate, NormMode::Normalized >( a, b );
    check_span_modes< AddMode::Accurate, NormMode::Unnormalized >( a, b );
    check_span_modes< AddMode::Sloppy, NormMode::Normalized >( a, b );
    check_span_modes< AddMode::Sloppy, NormMode::Unnormalized >( a, b );

    SoA product( n ), quotient( n );
    mul( product.span(), a.span(), b.span() );
    div( quotient.span(), a.span(), b.span() );
    SoA in_place = a;
    mul( in_place.span(), in_place.span(), b.span() );
    div( in_place.span(), in_place.span(), b.span() );
    bool defaults = true, round_trip = true;
    for( std::size_t i = 0; i < n; ++i ) {
        defaults &= identical( product.span().load( i ), a.span().load( i ) * b.span().load( i ) );
        defaults &= identical( quotient.span().load( i ), a.span().load( i ) / b.span().load( i ) );
        round_trip &= identical( in_place.span().load( i ), product.span().load( i ) / b.span().load( i ) );
    }
    check( defaults, "span mul/div defaults match operator* and operator/" );
    check( round_trip, "span mul/div in place" );

    SoA shorter( n - 1 );
    bool threw = false;
    try {
        mul( shorter.span(), a.span(), b.span() );
    }
    catch( const std::invalid_argument& ) {
        threw = true;
    }
    check( threw, "span mul with mismatched sizes throws" );
}

int main()
{
    std::mt19937_64 rng( 42 );
    mpfr_set_default_prec( MPFR_PREC );

    test_conj_real_imag_norm( rng );
    test_div( rng );
    test_span( rng );

    mpfr_free_cache();
    return failures == 0 ? 0 : 1;
}
