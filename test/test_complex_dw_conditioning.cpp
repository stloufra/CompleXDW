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

    mpfr_set_default_prec(MPFR_PREC);

    const int N_ITERATIONS = 100000;
    const double K_min = 10.0;
    const double K_max = 1e50;

    std::vector<TestResult> results(N_ITERATIONS);

    mpfr_t ar, ai, br, bi, cr, ci;
    mpfr_init2(ar, MPFR_PREC);
    mpfr_init2(ai, MPFR_PREC);
    mpfr_init2(br, MPFR_PREC);
    mpfr_init2(bi, MPFR_PREC);
    mpfr_init2(cr, MPFR_PREC);
    mpfr_init2(ci, MPFR_PREC);

    size_t idx = 0;
    auto start_time = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < N_ITERATIONS; ++i) {
        double t = N_ITERATIONS > 1 ? static_cast<double>(i) / (N_ITERATIONS - 1) : 0.0; //if only one K
        double target_K = K_min * std::pow(K_max / K_min, t);
        
        if (std::isnan(target_K) || std::isinf(target_K) || target_K <= 0) {
            std::cerr << "Invalid K at iteration " << i << ", skipping" << '\n';
            continue;
        }
        
#ifdef __DEBUG__
        std::cout << "=== Testing with K = " << target_K << " ===" << '\n';
#endif
        
        mpfr_t K_mpfr, K_check;
        mpfr_init2(K_mpfr, MPFR_PREC);
        mpfr_init2(K_check, MPFR_PREC);
        mpfr_set_d(K_mpfr, target_K, MPFR_RNDN); //set to target
        
        if (!generate_abcd_mp(K_mpfr, ar, ai, br, bi, K_check, rng, 100000)) {
            std::cerr << "Failed to generate abcd for K = " << target_K << '\n';
            mpfr_clear(K_mpfr);
            mpfr_clear(K_check);
            continue;
        }
        
        double K_actual = mpfr_get_d(K_check, MPFR_RNDN);
        
        mpfr_clear(K_mpfr);
        mpfr_clear(K_check);

        double ar_h, ar_l, ai_h, ai_l;
        double br_h, br_l, bi_h, bi_l;
        mpfr_to_dw(ar, MPFR_RNDN, &ar_h, &ar_l);
        mpfr_to_dw(ai, MPFR_RNDN, &ai_h, &ai_l);
        mpfr_to_dw(br, MPFR_RNDN, &br_h, &br_l);
        mpfr_to_dw(bi, MPFR_RNDN, &bi_h, &bi_l);

        ComplexDouble<double> a(ar_h, ar_l, ai_h, ai_l);
        ComplexDouble<double> b(br_h, br_l, bi_h, bi_l);

#ifdef __DEBUG__
        std::cout << "\n=== DEBUG: After assigning numbers ===" << '\n';
        std::cout << "K = " << target_K << '\n';
        std::cout << "Calling print_mpfr_complex for a..." << '\n';
        std::cout << std::flush;
        print_mpfr_complex("a", ar, ai);
        std::cout << "a printed, calling for b..." << '\n';
        std::cout << std::flush;
        print_mpfr_complex("b", br, bi);
        std::cout << "b printed" << '\n';
        std::cout << std::flush;
        std::cout << "DW a = (" << ar_h << ", " << ar_l << ") + i(" << ai_h << ", " << ai_l << ")" << '\n';
        std::cout << "DW b = (" << br_h << ", " << br_l << ") + i(" << bi_h << ", " << bi_l << ")" << '\n';
        std::cout << std::flush;
#endif

        ComplexDouble<double> c_norm = a * b;

#ifdef __DEBUG__
        std::cout << "\n=== DEBUG: After a * b ===" << '\n';
        std::cout << "c_norm = (" << c_norm.re_h() << ", " << c_norm.re_l() << ") + i("
                  << c_norm.im_h() << ", " << c_norm.im_l() << ")" << '\n';
#endif

        ComplexDouble<double> c_fast = mul_fast(a, b);

#ifdef __DEBUG__
        std::cout << "\n=== DEBUG: After mul_fast ===" << '\n';
        std::cout << "c_fast = (" << c_fast.re_h() << ", " << c_fast.re_l() << ") + i("
                  << c_fast.im_h() << ", " << c_fast.im_l() << ")" << '\n';
#endif

        mpfr_complex_mul(ar, ai, br, bi, cr, ci, MPFR_RNDN);

#ifdef __DEBUG__
        std::cout << "\n=== DEBUG: After mpfr_complex_mul ===" << '\n';
        print_mpfr_complex("cr, ci", cr, ci);
#endif

        double ref_re_h, ref_re_l, ref_im_h, ref_im_l;
        mpfr_to_dw(cr, MPFR_RNDN, &ref_re_h, &ref_re_l);
        mpfr_to_dw(ci, MPFR_RNDN, &ref_im_h, &ref_im_l);

        double err_norm = relative_error(cr, c_norm.re_h(), c_norm.re_l(), cr, MPFR_RNDN);
        double err_norm2 = relative_error(ci, c_norm.im_h(), c_norm.im_l(), ci, MPFR_RNDN);
        err_norm = std::max(err_norm, err_norm2);

        double err_fast = relative_error(cr, c_fast.re_h(), c_fast.re_l(), cr, MPFR_RNDN);
        double err_fast2 = relative_error(ci, c_fast.im_h(), c_fast.im_l(), ci, MPFR_RNDN);
        err_fast = std::max(err_fast, err_fast2);

        results[idx] = {err_norm, err_fast, K_actual,
                      ar_h, ar_l, ai_h, ai_l,
                      br_h, br_l, bi_h, bi_l,
                      c_norm.re_h(), c_norm.re_l(), c_norm.im_h(), c_norm.im_l(),
                      c_fast.re_h(), c_fast.re_l(), c_fast.im_h(), c_fast.im_l(),
                      ref_re_h, ref_re_l, ref_im_h, ref_im_l};

        idx++;

        if (idx % 10000 == 0) {
            std::cout << "Progress: " << idx << "/" << N_ITERATIONS << "\n";
        }
    }

    results.resize(idx);

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
