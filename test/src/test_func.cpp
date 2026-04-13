#include "test_func.h"
#include <random>
#include <algorithm>
#include <gmp.h>

void mpfr_to_dw(mpfr_t x, mpfr_rnd_t rnd, double* high, double* low) {
    mpfr_t hi, lo;
    mpfr_init2(hi, 53);
    mpfr_init2(lo, 53);

    mpfr_set(hi, x, rnd);
    mpfr_sub(lo, x, hi, rnd);

    *high = mpfr_get_d(hi, rnd);
    *low = mpfr_get_d(lo, rnd);

#ifdef __DEBUG__
    mpfr_t approx;
    mpfr_init2(approx, 1024);
    double re = relative_error(x, *high, *low, approx, rnd);
    if (re > 1e-30) {
    std::cout << "ERROR: mpfr_to_dw: relative error > 1e-30\n";
    throw std::runtime_error("mpfr_to_dw: relative error > 1e-30");
    }
    mpfr_clear(approx);
#endif

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

    mpfr_set_d(abs_err, approx_high, rnd);
    mpfr_add_d(abs_err, abs_err, approx_low, rnd);
    mpfr_sub(abs_err, abs_err, exact, rnd);
    mpfr_abs(abs_err, abs_err, rnd);

    mpfr_abs(abs_exact, exact, rnd);

    mpfr_div(rel_err, abs_err, abs_exact, rnd);

    double result = mpfr_get_d(rel_err, rnd);

    //if(result > 1) result=1;

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
    double max_err_acc_norm = 0.0, max_err_acc_un = 0.0, max_err_sloppy_un = 0.0;
    double sum_err_acc_norm = 0.0, sum_err_acc_un = 0.0, sum_err_sloppy_un = 0.0;
    size_t max_err_idx_acc_norm = 0, max_err_idx_acc_un = 0, max_err_idx_sloppy_un = 0;

    for (size_t i = 0; i < results.size(); ++i) {

        if (results[i].rel_err_acc_norm > max_err_acc_norm) {
            max_err_acc_norm = results[i].rel_err_acc_norm;
            max_err_idx_acc_norm = i;
        }
        if (results[i].rel_err_acc_un  > max_err_acc_un) {
            max_err_acc_un  = results[i].rel_err_acc_un;
            max_err_idx_acc_un  = i;
        }
	if (results[i].rel_err_sloppy_un  > max_err_sloppy_un) {
            max_err_sloppy_un = results[i].rel_err_sloppy_un;
            max_err_idx_sloppy_un = i;
        }

        sum_err_acc_norm += results[i].rel_err_acc_norm;
        sum_err_acc_un += results[i].rel_err_acc_un;
        sum_err_sloppy_un += results[i].rel_err_sloppy_un;

    }

    double avg_err_acc_norm = sum_err_acc_norm / results.size();
    double avg_err_acc_un = sum_err_acc_un / results.size();
    double avg_err_sloppy_un = sum_err_sloppy_un / results.size();

    print_separator();

    std::cout << "STATISTICS (Normalized Accurate Multiplication)\n";
    print_separator();
    std::cout << std::scientific << std::setprecision(6);
    std::cout << "  Average relative error: " << avg_err_acc_norm << "\n";
    std::cout << "  Maximum relative error: " << max_err_acc_norm<< "\n";
    std::cout << "  Occurred at iteration:  " << max_err_idx_acc_norm << "\n";

    std::cout << "STATISTICS (Unnormalized Accurate Multiplication)\n";
    print_separator();
    std::cout << std::scientific << std::setprecision(6);
    std::cout << "  Average relative error: " << avg_err_acc_un << "\n";
    std::cout << "  Maximum relative error: " << max_err_acc_un<< "\n";
    std::cout << "  Occurred at iteration:  " << max_err_idx_acc_un << "\n";

    print_separator();
    std::cout << "STATISTICS (Unnormalized Sloppy Multiplication)\n";
    print_separator();
    std::cout << "  Average relative error: " << avg_err_sloppy_un << "\n";
    std::cout << "  Maximum relative error: " << max_err_sloppy_un << "\n";
    std::cout << "  Occurred at iteration:  " << max_err_idx_sloppy_un << "\n";

    print_separator();

    std::cout << "WORST CASE (Normalized Accurate)\n";
    print_separator();
    const auto& wn = results[max_err_idx_acc_norm];
    std::cout << std::fixed << std::setprecision(17);
    std::cout << "  a = (" << wn.a_re_high << ", " << wn.a_re_low << ") + i(" 
              << wn.a_im_high << ", " << wn.a_im_low << ")\n";
    std::cout << "  b = (" << wn.b_re_high << ", " << wn.b_re_low << ") + i(" 
              << wn.b_im_high << ", " << wn.b_im_low << ")\n";
    std::cout << std::scientific << std::setprecision(MPFR_DISPLAY_PREC);
    std::cout << "  res = (" << wn.res_acc_norm_re_high << ", " << wn.res_acc_norm_re_low << ") + i(" 
              << wn.res_acc_norm_im_high << ", " << wn.res_acc_norm_im_low << ")\n";
    std::cout << "  ref = (" << wn.re_ref_high << ", " << wn.re_ref_low << ") + i(" 
              << wn.im_ref_high << ", " << wn.im_ref_low << ")\n";
    std::cout << "  error = " << wn.rel_err_acc_norm << "\n";
    print_separator();

    std::cout << "WORST CASE (Unnormalized Accurate)\n";
    print_separator();
    const auto& wu = results[max_err_idx_acc_un];
    std::cout << std::fixed << std::setprecision(17);
    std::cout << "  a = (" << wu.a_re_high << ", " << wu.a_re_low << ") + i(" 
              << wu.a_im_high << ", " << wu.a_im_low << ")\n";
    std::cout << "  b = (" << wu.b_re_high << ", " << wu.b_re_low << ") + i(" 
              << wu.b_im_high << ", " << wu.b_im_low << ")\n";
    std::cout << std::scientific << std::setprecision(MPFR_DISPLAY_PREC);
    std::cout << "  res = (" << wu.res_acc_un_re_high << ", " << wu.res_acc_un_re_low << ") + i(" 
              << wu.res_acc_un_im_high << ", " << wu.res_acc_un_im_low << ")\n";
    std::cout << "  ref = (" << wu.re_ref_high << ", " << wu.re_ref_low << ") + i(" 
              << wu.im_ref_high << ", " << wu.im_ref_low << ")\n";
    std::cout << "  error = " << wu.rel_err_acc_un << "\n";
    print_separator();

    std::cout << "WORST CASE (Unnormalized Sloppy)\n";
    print_separator();
    const auto& wf = results[max_err_idx_sloppy_un];
    std::cout << std::fixed << std::setprecision(17);
    std::cout << "  a = (" << wf.a_re_high << ", " << wf.a_re_low << ") + i(" 
              << wf.a_im_high << ", " << wf.a_im_low << ")\n";
    std::cout << "  b = (" << wf.b_re_high << ", " << wf.b_re_low << ") + i(" 
              << wf.b_im_high << ", " << wf.b_im_low << ")\n";
    std::cout << std::scientific << std::setprecision(MPFR_DISPLAY_PREC);
    std::cout << "  res = (" << wf.res_sloppy_un_re_high << ", " << wf.res_sloppy_un_re_low << ") + i(" 
              << wf.res_sloppy_un_im_high << ", " << wf.res_sloppy_un_im_low << ")\n";
    std::cout << "  ref = (" << wf.re_ref_high << ", " << wf.re_ref_low << ") + i(" 
              << wf.im_ref_high << ", " << wf.im_ref_low << ")\n";
    std::cout << "  error = " << wf.rel_err_sloppy_un << "\n";
    print_separator();
}

void save_results(const std::vector<TestResult>& results, const std::string& filename) {
    std::ofstream file(filename);
    file << "# Complex DW Multiplication Test Results\n";
    file << "# Iterations: " << results.size() << "\n";
    file << "# MPFR Precision: " << MPFR_PREC << " bits\n";
    file << "# Format: ar_h,ar_l,ai_h,ai_l,br_h,br_l,bi_h,bi_l,ref_re_h,ref_re_l,ref_im_h,ref_im_l,rel_err_acc_norm,rel_err_acc_un,rel_err_sloppy_un,K\n";
    
    file << std::scientific << std::setprecision(MPFR_DISPLAY_PREC);
    for (const auto& r : results) {
        file << r.a_re_high << "," << r.a_re_low << ","
             << r.a_im_high << "," << r.a_im_low << ","
             << r.b_re_high << "," << r.b_re_low << ","
             << r.b_im_high << "," << r.b_im_low << ","
             << r.re_ref_high << "," << r.re_ref_low << ","
             << r.im_ref_high << "," << r.im_ref_low << ","
             << r.rel_err_acc_norm << "," << r.rel_err_acc_un << "," << r.rel_err_sloppy_un << ","
             << r.K << "\n";
    }
    file.close();
    std::cout << "\nResults saved to: " << filename << "\n";
}

bool generate_abcd_mp(mpfr_t K, mpfr_t a, mpfr_t b, mpfr_t c, mpfr_t d, mpfr_t K_check, 
                      std::mt19937_64& rng, int max_tries) {
    //a = x_{real}, b= x_{imag}, c=y_{real}, d = y_{imag}
    //p = x_{imag}*y_{real} q = x_{real}*y_{imag}
    //p = a*d, q=b*c
    //K = |q|+|p|/|p+q|
    gmp_randstate_t state;
    gmp_randinit_mt(state);
    gmp_randseed_ui(state, rng());

    mpfr_t q, p, denom, nom, fabs_q, fabs_p, K_calc;
    mpfr_inits2(MPFR_PREC, q, p, denom, nom, fabs_q, fabs_p, K_calc, (mpfr_ptr)0);

    mpfr_t K_minus_1, K_plus_1;
    mpfr_init2(K_minus_1, MPFR_PREC);
    mpfr_init2(K_plus_1, MPFR_PREC);
    mpfr_sub_ui(K_minus_1, K, 1, MPFR_RNDN);
    mpfr_add_ui(K_plus_1, K, 1, MPFR_RNDN);

    double K_double = mpfr_get_d(K, MPFR_RNDN);
    double K_mag = std::pow(10, std::floor(std::log10(K_double)));

    constexpr double u64_max = static_cast<double>(std::numeric_limits<uint64_t>::max());
    double rnd_exp_1 = K_mag * (rng() / u64_max);
    double rnd_exp_2 = K_mag * (rng() / u64_max);
    double rnd_exp_3 = K_mag * (rng() / u64_max);
    
    for (int attempt = 0; attempt < max_tries; ++attempt) {

        mpfr_urandom(a, state, MPFR_RNDN);
        mpfr_mul_d(a, a, rnd_exp_1, MPFR_RNDN);
        mpfr_sub_d(a, a, rnd_exp_1/2, MPFR_RNDN);

        mpfr_urandom(c, state, MPFR_RNDN);
        mpfr_mul_d(c, c, rnd_exp_2, MPFR_RNDN);
        mpfr_sub_d(c, c, rnd_exp_2/2, MPFR_RNDN);

        mpfr_urandom(d, state, MPFR_RNDN);
        mpfr_mul_d(d, d, rnd_exp_3, MPFR_RNDN);
        mpfr_sub_d(d, d, rnd_exp_3/2, MPFR_RNDN);

#ifdef __DEBUG__
        std::cout << "a = " << mpfr_get_d(a, MPFR_RNDN) << '\n';
        std::cout << "c = " << mpfr_get_d(c, MPFR_RNDN) << '\n';
        std::cout << "d = " << mpfr_get_d(d, MPFR_RNDN) << '\n';
#endif

        if (mpfr_zero_p(a) || mpfr_zero_p(c) || mpfr_zero_p(d)) {
            continue;
        }

        mpfr_mul(p, a, d, MPFR_RNDN);
        mpfr_abs(fabs_p, p, MPFR_RNDN);

        int i = rng() % 4;
	
#ifdef __DEBUG__
	std::cout << "++++++++++++++++++++++++++++++++++" << '\n';
	std::cout<< "i = " << i << '\n';
	std::cout << "++++++++++++++++++++++++++++++++++" << '\n';
#endif

        if (i == 0) { // q > 0, (p+q) > 0,  q = (|p| -Kp)/(K-1)
            mpfr_mul(q, K, p, MPFR_RNDN);
            mpfr_sub(q, fabs_p, q, MPFR_RNDN);
            mpfr_div(q, q, K_minus_1, MPFR_RNDN);

        } else if (i == 1) { // q > 0, (p+q) < 0,  q = (-|p| -Kp)/(K+1)
            mpfr_mul(q, K, p, MPFR_RNDN);
            mpfr_add(q, q, fabs_p, MPFR_RNDN); //-(|p|+Kp)
            mpfr_neg(q, q, MPFR_RNDN);
            mpfr_div(q, q, K_plus_1, MPFR_RNDN);

        } else if (i == 2) { // q < 0, (p+q) > 0,  q = (|p| -Kp)/(K+1)
            mpfr_mul(q, K, p, MPFR_RNDN);
            mpfr_sub(q, fabs_p, q, MPFR_RNDN);
            mpfr_div(q, q, K_plus_1, MPFR_RNDN);

        } else if (i == 3) { // q < 0, (p+q) < 0,  q = (-|p| -Kp)/(K-1)
            mpfr_mul(q, K, p, MPFR_RNDN);
            mpfr_add(q, q, fabs_p, MPFR_RNDN);
            mpfr_neg(q, q, MPFR_RNDN); //-(|p|+Kp)
            mpfr_div(q, q, K_minus_1, MPFR_RNDN);
        }

        //check if valid with previous assumptions
        bool valid = false;
        if (i == 0) {
            mpfr_add(denom, q, p, MPFR_RNDN);
            if (mpfr_sgn(q) > 0 && mpfr_sgn(denom) > 0) valid = true;
        } else if (i == 1) {
            mpfr_add(denom, q, p, MPFR_RNDN);
            if (mpfr_sgn(q) > 0 && mpfr_sgn(denom) < 0) valid = true;
        } else if (i == 2) {
            mpfr_add(denom, q, p, MPFR_RNDN);
            if (mpfr_sgn(q) < 0 && mpfr_sgn(denom) > 0) valid = true;
        } else if (i == 3) {
            mpfr_add(denom, q, p, MPFR_RNDN);
            if (mpfr_sgn(q) < 0 && mpfr_sgn(denom) < 0) valid = true;
        }

        if (!valid) continue;

        //calculate d
        mpfr_div(b, q, c, MPFR_RNDN);

        //|q+p|
        mpfr_abs(denom, denom, MPFR_RNDN);

        //|q|
        mpfr_abs(fabs_q, q, MPFR_RNDN);

        if (mpfr_zero_p(denom)) {
            continue;
        }

        mpfr_add(nom, fabs_q, fabs_p, MPFR_RNDN);
        mpfr_div(K_calc, nom, denom, MPFR_RNDN);

        mpfr_t diff, rel_diff;
        mpfr_init2(diff, MPFR_PREC);
        mpfr_init2(rel_diff, MPFR_PREC);

        mpfr_sub(diff, K_calc, K, MPFR_RNDN);
        mpfr_abs(diff, diff, MPFR_RNDN);
        mpfr_div(rel_diff, diff, K, MPFR_RNDN);

        if (mpfr_cmp_d(rel_diff, 1e-9) < 0) {
            mpfr_set(K_check, K_calc, MPFR_RNDN);
#ifdef __DEBUG__
            std::cout << "SUCCESS! K_check = " << mpfr_get_d(K_check, MPFR_RNDN) << '\n';
            std::cout << "a = " << mpfr_get_d(a, MPFR_RNDN) << '\n';
            std::cout << "b = " << mpfr_get_d(b, MPFR_RNDN) << '\n';
            std::cout << "c = " << mpfr_get_d(c, MPFR_RNDN) << '\n';
            std::cout << "d = " << mpfr_get_d(d, MPFR_RNDN) << '\n';
#endif
            mpfr_clear(K_minus_1);
            mpfr_clear(K_plus_1);
            mpfr_clears(q, p, denom, nom, fabs_q, fabs_p, K_calc, diff, rel_diff, (mpfr_ptr)0);
            gmp_randclear(state);
            return true;
        }
        else{
#ifdef __DEBUG__
        std::cout << "Rel. dif. to high = " << mpfr_get_d(rel_diff, MPFR_RNDN) << '\n';
#endif
        }

        mpfr_clear(diff);
        mpfr_clear(rel_diff);
    }

#ifdef __DEBUG__
    std::cout << "FAILED to generate abcd after " << max_tries << " attempts" << '\n';
#endif

    mpfr_clear(K_minus_1);
    mpfr_clear(K_plus_1);
    mpfr_clears(q, p, denom, nom, fabs_q, fabs_p, K_calc, (mpfr_ptr)0);
    gmp_randclear(state);
    return false;
}

void print_mpfr(const char* name, mpfr_t x) {
    mpfr_exp_t exp;  // required exponent output
    char* str = mpfr_get_str(nullptr, &exp, 10, 0, x, MPFR_RNDN);

    std::cout << name << " = " << str << "e" << exp << '\n';

    mpfr_free_str(str);
}

void print_mpfr_complex(const char* name, mpfr_t re, mpfr_t im) {
    mpfr_exp_t re_exp, im_exp;

    char* re_str = mpfr_get_str(nullptr, &re_exp, 10, 0, re, MPFR_RNDN);
    char* im_str = mpfr_get_str(nullptr, &im_exp, 10, 0, im, MPFR_RNDN);

    std::cout << name << " = "
              << re_str << "e" << re_exp
              << " + i(" << im_str << "e" << im_exp << ")"
              << std::endl;

    mpfr_free_str(re_str);
    mpfr_free_str(im_str);
}
