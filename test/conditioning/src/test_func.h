#ifndef TEST_FUNC_H
#define TEST_FUNC_H

// MPFR helpers for the binned conditioning runs.

#include <random>

#include <mpfr.h>

#include "XDW.h"

constexpr int MPFR_PREC = 1024;
constexpr int MPFR_DISPLAY_PREC = 17;

// Rounds x to hi, then the remainder to lo.
void mpfr_to_dw(mpfr_t x, mpfr_rnd_t rnd, double* high, double* low);

// (cr, ci) = (ar + i ai) * (br + i bi).
void mpfr_complex_mul(mpfr_t ar, mpfr_t ai, mpfr_t br, mpfr_t bi,
                      mpfr_t cr, mpfr_t ci, mpfr_rnd_t rnd);

double relative_error(mpfr_t exact, double approx_high, double approx_low, mpfr_t approx_mpfr, mpfr_rnd_t rnd);

// Random a, b, c, d with (|a*d| + |b*c|) / |a*d + b*c| = K; false if max_tries attempts fail.
bool generate_abcd_mp(mpfr_t K, mpfr_t a, mpfr_t b, mpfr_t c, mpfr_t d, mpfr_t K_check,
                      std::mt19937_64& rng, int max_tries = 10000);

#endif
