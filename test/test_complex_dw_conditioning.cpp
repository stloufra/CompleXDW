#include <array>
#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <chrono>
#include <fstream>
#include <string>
#include <cmath>

#include <mpfr.h>

#include "../XDW.h"
#include "src/test_func.h"
#include "src/combo_mul.h"

using namespace XDW_ARTH;

struct ConditioningResult {
    double K;
    double ar_h, ar_l, ai_h, ai_l;
    double br_h, br_l, bi_h, bi_l;
    double ref_re_h, ref_re_l, ref_im_h, ref_im_l;
    std::array<double, N_COMBOS> rel_err;
    std::array<XDW<double>, N_COMBOS> result;
};

static void print_statistics_6(const std::vector<ConditioningResult>& results)
{
    for (int k = 0; k < N_COMBOS; ++k) {
        double max_err = 0.0, sum_err = 0.0;
        size_t max_idx = 0;
        for (size_t i = 0; i < results.size(); ++i) {
            sum_err += results[i].rel_err[k];
            if (results[i].rel_err[k] > max_err) {
                max_err = results[i].rel_err[k];
                max_idx = i;
            }
        }
        double avg_err = sum_err / results.size();

        print_separator();
        std::cout << "STATISTICS (" << COMBO_NAMES[k] << ")\n";
        print_separator();
        std::cout << std::scientific << std::setprecision(6);
        std::cout << "  Average relative error: " << avg_err << "\n";
        std::cout << "  Maximum relative error: " << max_err << "\n";
        std::cout << "  Occurred at iteration:  " << max_idx << " (K = " << results[max_idx].K << ")\n";

        const auto& w = results[max_idx];
        const auto& res = w.result[k];
        std::cout << std::fixed << std::setprecision(17);
        std::cout << "  a = (" << w.ar_h << ", " << w.ar_l << ") + i(" << w.ai_h << ", " << w.ai_l << ")\n";
        std::cout << "  b = (" << w.br_h << ", " << w.br_l << ") + i(" << w.bi_h << ", " << w.bi_l << ")\n";
        std::cout << std::scientific << std::setprecision(MPFR_DISPLAY_PREC);
        std::cout << "  res = (" << res.re_h() << ", " << res.re_l() << ") + i(" << res.im_h() << ", " << res.im_l() << ")\n";
        std::cout << "  ref = (" << w.ref_re_h << ", " << w.ref_re_l << ") + i(" << w.ref_im_h << ", " << w.ref_im_l << ")\n";
    }
    print_separator();
}

static void save_results_6(const std::vector<ConditioningResult>& results, const std::string& filename)
{
    std::ofstream file(filename);
    file << "# Complex DW Multiplication Test Results (all 6 AddMode/NormMode combinations)\n";
    file << "# Iterations: " << results.size() << "\n";
    file << "# MPFR Precision: " << MPFR_PREC << " bits\n";
    file << "# Format: ar_h,ar_l,ai_h,ai_l,br_h,br_l,bi_h,bi_l,ref_re_h,ref_re_l,ref_im_h,ref_im_l,";
    for (int k = 0; k < N_COMBOS; ++k) file << "rel_err_" << k << ",";
    file << "K\n";
    file << "# rel_err_0..5 = " << COMBO_NAMES[0] << ", " << COMBO_NAMES[1] << ", " << COMBO_NAMES[2]
         << ", " << COMBO_NAMES[3] << ", " << COMBO_NAMES[4] << ", " << COMBO_NAMES[5] << "\n";

    file << std::scientific << std::setprecision(MPFR_DISPLAY_PREC);
    for (const auto& r : results) {
        file << r.ar_h << "," << r.ar_l << "," << r.ai_h << "," << r.ai_l << ","
             << r.br_h << "," << r.br_l << "," << r.bi_h << "," << r.bi_l << ","
             << r.ref_re_h << "," << r.ref_re_l << "," << r.ref_im_h << "," << r.ref_im_l << ",";
        for (int k = 0; k < N_COMBOS; ++k) file << r.rel_err[k] << ",";
        file << r.K << "\n";
    }
    file.close();
    std::cout << "\nResults saved to: " << filename << "\n";
}

int main() {
    std::cout << std::setprecision(MPFR_DISPLAY_PREC);

    std::mt19937_64 rng(std::chrono::high_resolution_clock::now().time_since_epoch().count());

    mpfr_set_default_prec(MPFR_PREC);

    const int N_ITERATIONS = 100000;
    const double K_min = 10.0;
    const double K_max = 1e30;

    std::vector<ConditioningResult> results(N_ITERATIONS);

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

        mpfr_t K_mpfr, K_check;
        mpfr_init2(K_mpfr, MPFR_PREC);
        mpfr_init2(K_check, MPFR_PREC);
        mpfr_set_d(K_mpfr, target_K, MPFR_RNDN); //set to target

    //                a,  b,  c,  d
    //                ar*br-ai*bi -> ac-bd
    //                ai*br + ar*bi -> b*c + a*d
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

        XDW<double> a(ar_h, ar_l, ai_h, ai_l);
        XDW<double> b(br_h, br_l, bi_h, bi_l);

        std::array<XDW<double>, N_COMBOS> c = mul_all_combos(a, b);

        mpfr_complex_mul(ar, ai, br, bi, cr, ci, MPFR_RNDN);

        double ref_re_h, ref_re_l, ref_im_h, ref_im_l;
        mpfr_to_dw(cr, MPFR_RNDN, &ref_re_h, &ref_re_l);
        mpfr_to_dw(ci, MPFR_RNDN, &ref_im_h, &ref_im_l);

        std::array<double, N_COMBOS> err;
        for (int k = 0; k < N_COMBOS; ++k) {
            double err_re = relative_error(cr, c[k].re_h(), c[k].re_l(), cr, MPFR_RNDN);
            double err_im = relative_error(ci, c[k].im_h(), c[k].im_l(), ci, MPFR_RNDN);
            err[k] = std::max(err_re, err_im);
        }

        results[idx] = {K_actual,
                      ar_h, ar_l, ai_h, ai_l,
                      br_h, br_l, bi_h, bi_l,
                      ref_re_h, ref_re_l, ref_im_h, ref_im_l,
                      err, c};

        idx++;

        if (idx % 10000 == 0) {
            std::cout << "Progress: " << idx << "/" << N_ITERATIONS << "\n";
        }
    }

    results.resize(idx);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    print_header();
    print_statistics_6(results);

    std::cout << "\nExecution time: " << duration.count() << " ms\n";

    std::string filename = "res/test_results.csv";
    save_results_6(results, filename);

    mpfr_clear(ar);
    mpfr_clear(ai);
    mpfr_clear(br);
    mpfr_clear(bi);
    mpfr_clear(cr);
    mpfr_clear(ci);

    mpfr_free_cache();

    return 0;
}
