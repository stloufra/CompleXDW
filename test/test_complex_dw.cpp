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

using namespace XDW_ARTH;

constexpr int MPFR_PREC = 128;
constexpr int N_ITERATIONS = 10000000;
constexpr int MPFR_DISPLAY_PREC = 40;

struct TestResult {
    double rel_err_norm;
    double rel_err_fast;
    double a_re_high;
    double a_re_low;
    double a_im_high;
    double a_im_low;
    double b_re_high;
    double b_re_low;
    double b_im_high;
    double b_im_low;
    double res_norm_re_high;
    double res_norm_re_low;
    double res_norm_im_high;
    double res_norm_im_low;
    double res_fast_re_high;
    double res_fast_re_low;
    double res_fast_im_high;
    double res_fast_im_low;
    double re_ref_high;
    double re_ref_low;
    double im_ref_high;
    double im_ref_low;
};

void mpfr_to_dw(mpfr_t x, mpfr_rnd_t rnd, double* high, double* low) {
    mpfr_t hi, lo;
    mpfr_init2(hi, 53);
    mpfr_init2(lo, 53);

    mpfr_set(hi, x, rnd);
    mpfr_sub(lo, x, hi, rnd);

    *high = mpfr_get_d(hi, rnd);
    *low = mpfr_get_d(lo, rnd);

    mpfr_clear(hi);
    mpfr_clear(lo);
}

void mpfr_complex_mul(mpfr_t ar, mpfr_t ai, mpfr_t br, mpfr_t bi, 
                       mpfr_t cr, mpfr_t ci, mpfr_rnd_t rnd) {
    mpfr_t tr, ti, tr2, ti2;
    mpfr_init2(tr, MPFR_PREC);
    mpfr_init2(ti, MPFR_PREC);
    mpfr_init2(tr2, MPFR_PREC);
    mpfr_init2(ti2, MPFR_PREC);

    mpfr_mul(tr, ar, br, rnd);
    mpfr_mul(ti, ai, bi, rnd);
    mpfr_sub(cr, tr, ti, rnd);

    mpfr_mul(tr2, ar, bi, rnd);
    mpfr_mul(ti2, ai, br, rnd);
    mpfr_add(ci, tr2, ti2, rnd);

    mpfr_clear(tr);
    mpfr_clear(ti);
    mpfr_clear(tr2);
    mpfr_clear(ti2);
}

double relative_error(mpfr_t exact, double approx_high, double approx_low, mpfr_t approx_mpfr, mpfr_rnd_t rnd) {
    mpfr_t abs_err, abs_exact, rel_err;
    mpfr_init2(abs_err, MPFR_PREC);
    mpfr_init2(abs_exact, MPFR_PREC);
    mpfr_init2(rel_err, MPFR_PREC);

    mpfr_set_d(abs_err, approx_low, rnd);
    mpfr_add_d(abs_err, abs_err, approx_high, rnd);
    mpfr_sub(abs_err, abs_err, exact, rnd);
    mpfr_abs(abs_err, abs_err, rnd);

    mpfr_abs(abs_exact, exact, rnd);

    mpfr_div(rel_err, abs_err, abs_exact, rnd);

    double result = mpfr_get_d(rel_err, rnd);

    mpfr_clear(abs_err);
    mpfr_clear(abs_exact);
    mpfr_clear(rel_err);

    return result;
}

void print_separator() {
    std::cout << std::string(80, '=') << "\n";
}

void print_header() {
    print_separator();
    std::cout << "Complex DW Multiplication Test Results\n";
    std::cout << "Iterations: " << N_ITERATIONS << "\n";
    std::cout << "MPFR Precision: " << MPFR_PREC << " bits (~" << MPFR_PREC * 0.301 << " digits)\n";
    print_separator();
}

void print_statistics(const std::vector<TestResult>& results) {
    double max_err_norm = 0.0, max_err_fast = 0.0;
    double sum_err_norm = 0.0, sum_err_fast = 0.0;
    size_t max_err_idx_norm = 0, max_err_idx_fast = 0;

    for (size_t i = 0; i < results.size(); ++i) {
        if (results[i].rel_err_norm > max_err_norm) {
            max_err_norm = results[i].rel_err_norm;
            max_err_idx_norm = i;
        }
        if (results[i].rel_err_fast > max_err_fast) {
            max_err_fast = results[i].rel_err_fast;
            max_err_idx_fast = i;
        }
        sum_err_norm += results[i].rel_err_norm;
        sum_err_fast += results[i].rel_err_fast;
    }

    double avg_err_norm = sum_err_norm / results.size();
    double avg_err_fast = sum_err_fast / results.size();

    print_separator();
    std::cout << "STATISTICS (Normalized Multiplication *)\n";
    print_separator();
    std::cout << std::scientific << std::setprecision(6);
    std::cout << "  Average relative error: " << avg_err_norm << "\n";
    std::cout << "  Maximum relative error: " << max_err_norm << "\n";
    std::cout << "  Occurred at iteration:  " << max_err_idx_norm << "\n";

    print_separator();
    std::cout << "STATISTICS (Fast Multiplication mul_fast)\n";
    print_separator();
    std::cout << "  Average relative error: " << avg_err_fast << "\n";
    std::cout << "  Maximum relative error: " << max_err_fast << "\n";
    std::cout << "  Occurred at iteration:  " << max_err_idx_fast << "\n";

    print_separator();
    std::cout << "WORST CASE (Normalized *)\n";
    print_separator();
    const auto& wn = results[max_err_idx_norm];
    std::cout << std::fixed << std::setprecision(17);
    std::cout << "  a = (" << wn.a_re_high << ", " << wn.a_re_low << ") + i(" 
              << wn.a_im_high << ", " << wn.a_im_low << ")\n";
    std::cout << "  b = (" << wn.b_re_high << ", " << wn.b_re_low << ") + i(" 
              << wn.b_im_high << ", " << wn.b_im_low << ")\n";
    std::cout << std::scientific << std::setprecision(MPFR_DISPLAY_PREC);
    std::cout << "  res = (" << wn.res_norm_re_high << ", " << wn.res_norm_re_low << ") + i(" 
              << wn.res_norm_im_high << ", " << wn.res_norm_im_low << ")\n";
    std::cout << "  ref = (" << wn.re_ref_high << ", " << wn.re_ref_low << ") + i(" 
              << wn.im_ref_high << ", " << wn.im_ref_low << ")\n";
    std::cout << "  error = " << wn.rel_err_norm << "\n";

    print_separator();
    std::cout << "WORST CASE (Fast mul_fast)\n";
    print_separator();
    const auto& wf = results[max_err_idx_fast];
    std::cout << std::fixed << std::setprecision(17);
    std::cout << "  a = (" << wf.a_re_high << ", " << wf.a_re_low << ") + i(" 
              << wf.a_im_high << ", " << wf.a_im_low << ")\n";
    std::cout << "  b = (" << wf.b_re_high << ", " << wf.b_re_low << ") + i(" 
              << wf.b_im_high << ", " << wf.b_im_low << ")\n";
    std::cout << std::scientific << std::setprecision(MPFR_DISPLAY_PREC);
    std::cout << "  res = (" << wf.res_fast_re_high << ", " << wf.res_fast_re_low << ") + i(" 
              << wf.res_fast_im_high << ", " << wf.res_fast_im_low << ")\n";
    std::cout << "  ref = (" << wf.re_ref_high << ", " << wf.re_ref_low << ") + i(" 
              << wf.im_ref_high << ", " << wf.im_ref_low << ")\n";
    std::cout << "  error = " << wf.rel_err_fast << "\n";

    print_separator();
}

void save_results(const std::vector<TestResult>& results, const std::string& filename) {
    std::ofstream file(filename);
    file << "# Complex DW Multiplication Test Results\n";
    file << "# Iterations: " << N_ITERATIONS << "\n";
    file << "# MPFR Precision: " << MPFR_PREC << " bits\n";
    file << "# Format: a_re_h,a_re_l,a_im_h,a_im_l,b_re_h,b_re_l,b_im_h,b_im_l,ref_re_h,ref_re_l,ref_im_h,ref_im_l,rel_err_norm,rel_err_fast\n";
    
    file << std::scientific << std::setprecision(MPFR_DISPLAY_PREC);
    for (const auto& r : results) {
        file << r.a_re_high << "," << r.a_re_low << ","
             << r.a_im_high << "," << r.a_im_low << ","
             << r.b_re_high << "," << r.b_re_low << ","
             << r.b_im_high << "," << r.b_im_low << ","
             << r.re_ref_high << "," << r.re_ref_low << ","
             << r.im_ref_high << "," << r.im_ref_low << ","
             << r.rel_err_norm << "," << r.rel_err_fast << "\n";
    }
    file.close();
    std::cout << "\nResults saved to: " << filename << "\n";
}

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
