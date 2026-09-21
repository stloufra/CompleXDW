#include <algorithm>
#include <iostream>
#include <random>

#include <mpfr.h>

#include "../ComplexDouble.h"
#include "src/test_func.h"

constexpr int SAMPLES = 20000;
constexpr double UNIT_ROUNDOFF_SQUARED = 0x1p-106;
constexpr double NORM_ERROR_BOUND = 7.0 * UNIT_ROUNDOFF_SQUARED;

static int failures = 0;

static void check( bool condition, const char* what )
{
    if( condition )
        return;
    std::cerr << "FAILED: " << what << '\n';
    ++failures;
}

static bool has_parts( const ComplexDouble< double >& z, double reh, double rel, double imh, double iml )
{
    return z.re_h() == reh && z.re_l() == rel && z.im_h() == imh && z.im_l() == iml;
}

static void dw_to_mpfr( mpfr_t out, double high, double low )
{
    mpfr_set_d( out, high, MPFR_RNDN );
    mpfr_add_d( out, out, low, MPFR_RNDN );
}

int main()
{
    std::mt19937_64 rng( 42 );
    mpfr_set_default_prec( MPFR_PREC );

    double worst_norm_error = 0.0;

    for( int i = 0; i < SAMPLES; ++i ) {
        const auto z = generate_random_dw_single( rng );

        check( has_parts( conj( z ), z.re_h(), z.re_l(), -z.im_h(), -z.im_l() ), "conj" );
        check( has_parts( real( z ), z.re_h(), z.re_l(), 0.0, 0.0 ), "real" );
        check( has_parts( imag( z ), z.im_h(), z.im_l(), 0.0, 0.0 ), "imag" );

        const auto n = norm( z );
        check( n.im_h() == 0.0 && n.im_l() == 0.0, "norm has zero imaginary part" );

        mpfr_t re, im, reference, term;
        mpfr_inits2( MPFR_PREC, re, im, reference, term, (mpfr_ptr) nullptr );
        dw_to_mpfr( re, z.re_h(), z.re_l() );
        dw_to_mpfr( im, z.im_h(), z.im_l() );
        mpfr_mul( reference, re, re, MPFR_RNDN );
        mpfr_mul( term, im, im, MPFR_RNDN );
        mpfr_add( reference, reference, term, MPFR_RNDN );

        worst_norm_error = std::max( worst_norm_error, relative_error( reference, n.re_h(), n.re_l(), reference, MPFR_RNDN ) );
        mpfr_clears( re, im, reference, term, (mpfr_ptr) nullptr );
    }

    check( worst_norm_error <= NORM_ERROR_BOUND, "norm relative error within 7u^2" );
    std::cout << "norm worst relative error / u^2 = " << worst_norm_error / UNIT_ROUNDOFF_SQUARED << '\n';

    mpfr_free_cache();
    return failures == 0 ? 0 : 1;
}
