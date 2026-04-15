#ifndef TEST_FUNC_H
#define TEST_FUNC_H

#include <iostream>
#include <iomanip>
#include <vector>
#include <array>
#include <string>
#include <fstream>
#include <random>
#include <cmath>

#include <mpfr.h>

#include "ComplexDouble.h"

constexpr int MPFR_PREC = 1024;
constexpr int N_ITERATIONS = 10000000;
constexpr int MPFR_DISPLAY_PREC = 17;

struct TestResult {
    double rel_err_acc_norm;
    double rel_err_acc_un;
    double rel_err_sloppy_un;
    double K;
    double a_re_high;
    double a_re_low;
    double a_im_high;
    double a_im_low;
    double b_re_high;
    double b_re_low;
    double b_im_high;
    double b_im_low;
    double res_acc_norm_re_high;
    double res_acc_norm_re_low;
    double res_acc_norm_im_high;
    double res_acc_norm_im_low;
    double res_acc_un_re_high;
    double res_acc_un_re_low;
    double res_acc_un_im_high;
    double res_acc_un_im_low;
    double res_sloppy_un_re_high;
    double res_sloppy_un_re_low;
    double res_sloppy_un_im_high;
    double res_sloppy_un_im_low;
    double re_ref_high;
    double re_ref_low;
    double im_ref_high;
    double im_ref_low;
};

void mpfr_to_dw(mpfr_t x, mpfr_rnd_t rnd, double* high, double* low);

void mpfr_complex_mul(mpfr_t ar, mpfr_t ai, mpfr_t br, mpfr_t bi, 
                      mpfr_t cr, mpfr_t ci, mpfr_rnd_t rnd);

ComplexDouble<double> generate_random_dw_single(std::mt19937_64& rng);

template<size_t N>
void generate_random_dw_complex(std::array<ComplexDouble<double>, N>& a,
                                 std::array<ComplexDouble<double>, N>& b,
                                 std::mt19937_64& rng) {
    for (size_t i = 0; i < N; ++i) {
        a[i] = generate_random_dw_single(rng);
        b[i] = generate_random_dw_single(rng);
    }
}

double relative_error(mpfr_t exact, double approx_high, double approx_low, mpfr_t approx_mpfr, mpfr_rnd_t rnd);

void print_separator();

void print_header();

void print_statistics(const std::vector<TestResult>& results);

void save_results(const std::vector<TestResult>& results, const std::string& filename);

void print_mpfr(const char* name, mpfr_t x);

void print_mpfr_complex(const char* name, mpfr_t re, mpfr_t im);

bool generate_abcd_mp(mpfr_t K, mpfr_t a, mpfr_t b, mpfr_t c, mpfr_t d, mpfr_t K_check,
                     std::mt19937_64& rng, int max_tries = 10000);

#endif
