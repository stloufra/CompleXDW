#include "test_func.h"

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
    file << "# Format: ref_re_h,ref_re_l,ref_im_h,ref_im_l,rel_err_norm,rel_err_fast\n";
    
    file << std::scientific << std::setprecision(MPFR_DISPLAY_PREC);
    for (const auto& r : results) {
        file << r.re_ref_high << "," << r.re_ref_low << ","
             << r.im_ref_high << "," << r.im_ref_low << ","
             << r.rel_err_norm << "," << r.rel_err_fast << "\n";
    }
    file.close();
    std::cout << "\nResults saved to: " << filename << "\n";
}
