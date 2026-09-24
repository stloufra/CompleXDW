#ifndef BINNED_OPS_H
#define BINNED_OPS_H

// The operations the binned runs sweep. Errors are measured against the exact result of the DW
// inputs themselves (not of the higher-precision numbers they were rounded from), and are shared
// by the run and the replay so a stored worst case reproduces its stored error bit-for-bit.

#include <algorithm>
#include <array>

#include "test_func.h"
#include "combo_mul.h"
#include "combo_div.h"
#include "binned_results.h"

inline Args dw_args(mpfr_ptr ar, mpfr_ptr ai, mpfr_ptr br, mpfr_ptr bi)
{
    Args x;
    mpfr_to_dw(ar, MPFR_RNDN, &x.ar_h, &x.ar_l);
    mpfr_to_dw(ai, MPFR_RNDN, &x.ai_h, &x.ai_l);
    mpfr_to_dw(br, MPFR_RNDN, &x.br_h, &x.br_l);
    mpfr_to_dw(bi, MPFR_RNDN, &x.bi_h, &x.bi_l);
    return x;
}

template <std::size_t N>
std::array<double, N> max_component_errors(const std::array<ComplexDouble<double>, N>& c, mpfr_t re, mpfr_t im)
{
    std::array<double, N> err;
    for (std::size_t k = 0; k < N; ++k) {
        double err_re = relative_error(re, c[k].re_h(), c[k].re_l(), re, MPFR_RNDN);
        double err_im = relative_error(im, c[k].im_h(), c[k].im_l(), im, MPFR_RNDN);
        err[k] = std::max(err_re, err_im);
    }
    return err;
}

struct MulOp {
    static constexpr int N_COMBOS = ::N_COMBOS;
    static constexpr const auto& COMBO_NAMES = ::COMBO_NAMES;
    static constexpr const char* NAME = "multiplication";
    static constexpr const char* SYMBOL = "*";
    static constexpr const char* K_DEFINITION = "(|ar*bi| + |ai*br|) / |ar*bi + ai*br| (imaginary part of a*b)";
    static constexpr const char* RESULTS_FILE = "res/binned_results_mul.csv";
    static constexpr const char* LOG_FILE = "res/binned_log_mul.txt";

    // generate_abcd_mp conditions a*d + b*c = ar*bi + ai*br.
    static Args make_args(mpfr_ptr a, mpfr_ptr b, mpfr_ptr c, mpfr_ptr d) { return dw_args(a, b, c, d); }

    static double conditioning(const Args& x)
    {
        mpfr_t ar, ai, br, bi, p, q;
        mpfr_inits2(MPFR_PREC, ar, ai, br, bi, p, q, (mpfr_ptr) nullptr);
        args_to_mpfr(x, ar, ai, br, bi);
        mpfr_mul(p, ar, bi, MPFR_RNDN);
        mpfr_mul(q, ai, br, MPFR_RNDN);
        double K = sum_conditioning(p, q);
        mpfr_clears(ar, ai, br, bi, p, q, (mpfr_ptr) nullptr);
        return K;
    }

    static std::array<double, N_COMBOS> errors(const Args& x)
    {
        mpfr_t ar, ai, br, bi, cr, ci;
        mpfr_inits2(MPFR_PREC, ar, ai, br, bi, cr, ci, (mpfr_ptr) nullptr);
        args_to_mpfr(x, ar, ai, br, bi);
        mpfr_complex_mul(ar, ai, br, bi, cr, ci, MPFR_RNDN);
        auto err = max_component_errors(mul_all_combos(ComplexDouble<double>(x.ar_h, x.ar_l, x.ai_h, x.ai_l),
                                                       ComplexDouble<double>(x.br_h, x.br_l, x.bi_h, x.bi_l)),
                                        cr, ci);
        mpfr_clears(ar, ai, br, bi, cr, ci, (mpfr_ptr) nullptr);
        return err;
    }
};

// a / b = ((ar*br + ai*bi) + i(ai*br - ar*bi)) / (br^2 + bi^2). The denominator never cancels, so
// the conditioning is that of the numerator: the worse of its two components, since the error of
// each component of the quotient follows the conditioning of its own numerator.
struct DivOp {
    static constexpr int N_COMBOS = N_DIV_COMBOS;
    static constexpr const auto& COMBO_NAMES = DIV_COMBO_NAMES;
    static constexpr const char* NAME = "division";
    static constexpr const char* SYMBOL = "/";
    static constexpr const char* K_DEFINITION =
        "max((|ar*br| + |ai*bi|) / |ar*br + ai*bi|, (|ai*br| + |ar*bi|) / |ai*br - ar*bi|) (numerator of a/b)";
    static constexpr const char* RESULTS_FILE = "res/binned_results_div.csv";
    static constexpr const char* LOG_FILE = "res/binned_log_div.txt";

    // Dividing by (c - id) makes the imaginary numerator b*c + a*d, the sum generate_abcd_mp conditions.
    static Args make_args(mpfr_ptr a, mpfr_ptr b, mpfr_ptr c, mpfr_ptr d)
    {
        Args x = dw_args(a, b, c, d);
        x.bi_h = -x.bi_h;
        x.bi_l = -x.bi_l;
        return x;
    }

    static double conditioning(const Args& x)
    {
        mpfr_t ar, ai, br, bi, p, q;
        mpfr_inits2(MPFR_PREC, ar, ai, br, bi, p, q, (mpfr_ptr) nullptr);
        args_to_mpfr(x, ar, ai, br, bi);
        mpfr_mul(p, ar, br, MPFR_RNDN);
        mpfr_mul(q, ai, bi, MPFR_RNDN);
        double K_re = sum_conditioning(p, q);
        mpfr_mul(p, ai, br, MPFR_RNDN);
        mpfr_mul(q, ar, bi, MPFR_RNDN);
        mpfr_neg(q, q, MPFR_RNDN);
        double K_im = sum_conditioning(p, q);
        mpfr_clears(ar, ai, br, bi, p, q, (mpfr_ptr) nullptr);
        return std::max(K_re, K_im);
    }

    static std::array<double, N_COMBOS> errors(const Args& x)
    {
        mpfr_t ar, ai, br, bi, t, denom, qr, qi;
        mpfr_inits2(MPFR_PREC, ar, ai, br, bi, t, denom, qr, qi, (mpfr_ptr) nullptr);
        args_to_mpfr(x, ar, ai, br, bi);
        mpfr_sqr(denom, br, MPFR_RNDN);
        mpfr_sqr(t, bi, MPFR_RNDN);
        mpfr_add(denom, denom, t, MPFR_RNDN);
        mpfr_mul(qr, ar, br, MPFR_RNDN);
        mpfr_mul(t, ai, bi, MPFR_RNDN);
        mpfr_add(qr, qr, t, MPFR_RNDN);
        mpfr_div(qr, qr, denom, MPFR_RNDN);
        mpfr_mul(qi, ai, br, MPFR_RNDN);
        mpfr_mul(t, ar, bi, MPFR_RNDN);
        mpfr_sub(qi, qi, t, MPFR_RNDN);
        mpfr_div(qi, qi, denom, MPFR_RNDN);
        auto err = max_component_errors(div_all_combos(ComplexDouble<double>(x.ar_h, x.ar_l, x.ai_h, x.ai_l),
                                                       ComplexDouble<double>(x.br_h, x.br_l, x.bi_h, x.bi_l)),
                                        qr, qi);
        mpfr_clears(ar, ai, br, bi, t, denom, qr, qi, (mpfr_ptr) nullptr);
        return err;
    }
};

#endif  //BINNED_OPS_H
