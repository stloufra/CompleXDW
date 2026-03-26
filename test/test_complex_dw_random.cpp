#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <chrono>
#include <fstream>
#include <string>
#include <cmath>

#include <mpfr.h>

#include "../ComplexDouble.h"
#include "src/test_func.h"

using namespace XDW_ARTH;

int main() {
    std::cout << std::setprecision(MPFR_DISPLAY_PREC);
    
    std::mt19937_64 rng(std::chrono::high_resolution_clock::now().time_since_epoch().count());
    std::uniform_real_distribution<double> dist_exp(-10.0, 10.0);
    std::uniform_real_distribution<double> dist_frac(0.0, 1.0);

    mpfr_set_default_prec(MPFR_PREC);

    std::vector<TestResult> results(N_ITERATIONS);

    mpfr_t ar, ai, br, bi, cr, ci;
    mpfr_init2(ar, MPFR_PREC);
    mpfr_init2(ai, MPFR_PREC);
    mpfr_init2(br, MPFR_PREC);
    mpfr_init2(bi, MPFR_PREC);
    mpfr_init2(cr, MPFR_PREC);
    mpfr_init2(ci, MPFR_PREC);

    auto start_time = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < N_ITERATIONS; ++i) {
        int exp1_re = static_cast<int>(dist_exp(rng));
        int exp1_im = static_cast<int>(dist_exp(rng));
        int exp2_re = static_cast<int>(dist_exp(rng));
        int exp2_im = static_cast<int>(dist_exp(rng));

        mpfr_set_d(ar, 1.0, MPFR_RNDN);
        mpfr_set_d(ai, 1.0, MPFR_RNDN);
        mpfr_set_d(br, 1.0, MPFR_RNDN);
        mpfr_set_d(bi, 1.0, MPFR_RNDN);

        mpfr_mul_2si(ar, ar, exp1_re, MPFR_RNDN);
        mpfr_mul_2si(ai, ai, exp1_im, MPFR_RNDN);
        mpfr_mul_2si(br, br, exp2_re, MPFR_RNDN);
        mpfr_mul_2si(bi, bi, exp2_im, MPFR_RNDN);

        double rnd1_re = dist_frac(rng);
        double rnd1_im = dist_frac(rng);
        double rnd2_re = dist_frac(rng);
        double rnd2_im = dist_frac(rng);

        mpfr_add_d(ar, ar, rnd1_re, MPFR_RNDN);
        mpfr_add_d(ai, ai, rnd1_im, MPFR_RNDN);
        mpfr_add_d(br, br, rnd2_re, MPFR_RNDN);
        mpfr_add_d(bi, bi, rnd2_im, MPFR_RNDN);

        if (rng() % 2) mpfr_neg(ar, ar, MPFR_RNDN);
        if (rng() % 2) mpfr_neg(ai, ai, MPFR_RNDN);
        if (rng() % 2) mpfr_neg(br, br, MPFR_RNDN);
        if (rng() % 2) mpfr_neg(bi, bi, MPFR_RNDN);

        double ar_h, ar_l, ai_h, ai_l;
        double br_h, br_l, bi_h, bi_l;
        mpfr_to_dw(ar, MPFR_RNDN, &ar_h, &ar_l);
        mpfr_to_dw(ai, MPFR_RNDN, &ai_h, &ai_l);
        mpfr_to_dw(br, MPFR_RNDN, &br_h, &br_l);
        mpfr_to_dw(bi, MPFR_RNDN, &bi_h, &bi_l);

        ComplexDouble<double> a(ar_h, ar_l, ai_h, ai_l);
        ComplexDouble<double> b(br_h, br_l, bi_h, bi_l);

        ComplexDouble<double> c_norm = a * b;
        ComplexDouble<double> c_fast = mul_fast(a, b);

        mpfr_complex_mul(ar, ai, br, bi, cr, ci, MPFR_RNDN);

        double ref_re_h, ref_re_l, ref_im_h, ref_im_l;
        mpfr_to_dw(cr, MPFR_RNDN, &ref_re_h, &ref_re_l);
        mpfr_to_dw(ci, MPFR_RNDN, &ref_im_h, &ref_im_l);

        double err_norm = relative_error(cr, c_norm.re_h(), c_norm.re_l(), cr, MPFR_RNDN);
        double err_norm2 = relative_error(ci, c_norm.im_h(), c_norm.im_l(), ci, MPFR_RNDN);
        err_norm = std::max(err_norm, err_norm2);

        double err_fast = relative_error(cr, c_fast.re_h(), c_fast.re_l(), cr, MPFR_RNDN);
        double err_fast2 = relative_error(ci, c_fast.im_h(), c_fast.im_l(), ci, MPFR_RNDN);
        err_fast = std::max(err_fast, err_fast2);

        results[i] = {err_norm, err_fast,
                     ar_h, ar_l, ai_h, ai_l,
                     br_h, br_l, bi_h, bi_l,
                     c_norm.re_h(), c_norm.re_l(), c_norm.im_h(), c_norm.im_l(),
                     c_fast.re_h(), c_fast.re_l(), c_fast.im_h(), c_fast.im_l(),
                     ref_re_h, ref_re_l, ref_im_h, ref_im_l};

        if (i % 10000 == 0) {
            std::cout << "Progress: " << i << "/" << N_ITERATIONS << "\n";
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    print_header();
    print_statistics(results);
    
    std::cout << "\nExecution time: " << duration.count() << " ms\n";

    std::string filename = "test_results.csv";
    save_results(results, filename);

    mpfr_clear(ar);
    mpfr_clear(ai);
    mpfr_clear(br);
    mpfr_clear(bi);
    mpfr_clear(cr);
    mpfr_clear(ci);

    mpfr_free_cache();

    return 0;
}
