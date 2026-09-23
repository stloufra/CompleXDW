#ifndef BINNED_RESULTS_H
#define BINNED_RESULTS_H

// Checkpoint format shared by test_complex_dw_conditioning_binned.cpp (writes/resumes it) and
// test_complex_dw_conditioning_replay.cpp (reads a stored worst case back).

#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>

#include "test_func.h"
#include "combo_mul.h"

constexpr double K_MIN = 10.0;
constexpr double K_MAX = 1e30;
constexpr int NUM_BINS = 30;  // one per decade of K, [1e1,1e2), [1e2,1e3), ..., [1e30,1e31)
constexpr int ARG_COLS = 8;   // ar_h, ar_l, ai_h, ai_l, br_h, br_l, bi_h, bi_l
constexpr double U2 = 0x1p-106;  // u^2 for double, u = 2^-53

inline const std::string RESULTS_FILE = "res/binned_results.csv";
inline const std::string LOG_FILE = "res/binned_log.txt";

// The DW inputs (a, b) behind one particular result, so a worst case can be reproduced later
// with test_complex_dw_conditioning_replay.
struct Args {
    double ar_h = 0, ar_l = 0, ai_h = 0, ai_l = 0;
    double br_h = 0, br_l = 0, bi_h = 0, bi_l = 0;
};

// Per (bin, combo): stats of the relative error, and of the relative error normalized by K*u^2
// (nmax/nmean), which is what shows the constant C in "error <= C*K*u^2". The normalization is
// per sample, since K varies ~10x inside a bin and can't be recovered from the bin's error stats.
struct BinStats {
    long long count = 0;
    double min = 0.0, max = 0.0, mean = 0.0;
    double nmax = 0.0, nmean = 0.0;
    Args worst_args;  // inputs behind `max`

    void update(double x, double nx, const Args& args)
    {
        ++count;
        if (count == 1) { min = max = mean = x; nmax = nmean = nx; worst_args = args; return; }
        if (x < min) min = x;
        if (x > max) { max = x; worst_args = args; }
        if (nx > nmax) nmax = nx;
        mean += (x - mean) / static_cast<double>(count);
        nmean += (nx - nmean) / static_cast<double>(count);
    }
};

// K = (|ar*bi| + |ai*br|) / |ar*bi + ai*br|, the conditioning of the imaginary part of a*b (the
// definition generate_abcd_mp targets), evaluated on the DW inputs. Infinite if the sum is exactly 0.
inline double conditioning(const Args& x)
{
    mpfr_t ar, ai, br, bi, p, q, sum, abs_sum;
    mpfr_inits2(MPFR_PREC, ar, ai, br, bi, p, q, sum, abs_sum, (mpfr_ptr) nullptr);
    mpfr_set_d(ar, x.ar_h, MPFR_RNDN); mpfr_add_d(ar, ar, x.ar_l, MPFR_RNDN);
    mpfr_set_d(ai, x.ai_h, MPFR_RNDN); mpfr_add_d(ai, ai, x.ai_l, MPFR_RNDN);
    mpfr_set_d(br, x.br_h, MPFR_RNDN); mpfr_add_d(br, br, x.br_l, MPFR_RNDN);
    mpfr_set_d(bi, x.bi_h, MPFR_RNDN); mpfr_add_d(bi, bi, x.bi_l, MPFR_RNDN);
    mpfr_mul(p, ar, bi, MPFR_RNDN);
    mpfr_mul(q, ai, br, MPFR_RNDN);
    mpfr_add(sum, p, q, MPFR_RNDN);
    mpfr_abs(p, p, MPFR_RNDN); mpfr_abs(q, q, MPFR_RNDN); mpfr_abs(abs_sum, sum, MPFR_RNDN);
    mpfr_add(p, p, q, MPFR_RNDN);
    mpfr_div(p, p, abs_sum, MPFR_RNDN);
    double K = mpfr_get_d(p, MPFR_RNDN);
    mpfr_clears(ar, ai, br, bi, p, q, sum, abs_sum, (mpfr_ptr) nullptr);
    return K;
}

// Relative errors of all 6 mul<Add,Norm> combos for the product a*b, measured against the exact
// product of the DW inputs themselves (not of the higher-precision numbers they were rounded from).
// Shared by the run and the replay so a stored worst case reproduces its stored error bit-for-bit.
inline std::array<double, N_COMBOS> combo_errors(const Args& x)
{
    mpfr_t ar, ai, br, bi, cr, ci;
    mpfr_inits2(MPFR_PREC, ar, ai, br, bi, cr, ci, (mpfr_ptr) nullptr);
    mpfr_set_d(ar, x.ar_h, MPFR_RNDN); mpfr_add_d(ar, ar, x.ar_l, MPFR_RNDN);
    mpfr_set_d(ai, x.ai_h, MPFR_RNDN); mpfr_add_d(ai, ai, x.ai_l, MPFR_RNDN);
    mpfr_set_d(br, x.br_h, MPFR_RNDN); mpfr_add_d(br, br, x.br_l, MPFR_RNDN);
    mpfr_set_d(bi, x.bi_h, MPFR_RNDN); mpfr_add_d(bi, bi, x.bi_l, MPFR_RNDN);
    mpfr_complex_mul(ar, ai, br, bi, cr, ci, MPFR_RNDN);

    const auto c = mul_all_combos(ComplexDouble<double>(x.ar_h, x.ar_l, x.ai_h, x.ai_l),
                                  ComplexDouble<double>(x.br_h, x.br_l, x.bi_h, x.bi_l));
    std::array<double, N_COMBOS> err;
    for (int k = 0; k < N_COMBOS; ++k) {
        double err_re = relative_error(cr, c[k].re_h(), c[k].re_l(), cr, MPFR_RNDN);
        double err_im = relative_error(ci, c[k].im_h(), c[k].im_l(), ci, MPFR_RNDN);
        err[k] = std::max(err_re, err_im);
    }
    mpfr_clears(ar, ai, br, bi, cr, ci, (mpfr_ptr) nullptr);
    return err;
}

using BinRow = std::array<BinStats, N_COMBOS>;
using Bins = std::array<BinRow, NUM_BINS>;

inline int bin_index(double K)
{
    int b = static_cast<int>(std::floor(std::log10(K))) - 1;
    if (b < 0) b = 0;
    if (b >= NUM_BINS) b = NUM_BINS - 1;
    return b;
}

// Returns {total_attempted, total_valid, elapsed_seconds} from a prior checkpoint, or zeros if
// RESULTS_FILE doesn't exist. Fills `bins` with what it recovers. Throws (std::invalid_argument /
// std::out_of_range) on a malformed row, e.g. a checkpoint from before the worst-case args columns.
inline std::array<long long, 3> load_checkpoint(Bins& bins)
{
    std::array<long long, 3> totals = {0, 0, 0};
    std::ifstream in(RESULTS_FILE);
    if (!in) return totals;

    std::string line;
    while (std::getline(in, line)) {
        if (line.rfind("# total_attempted=", 0) == 0) {
            long long attempted = 0, valid = 0, elapsed = 0;
            std::sscanf(line.c_str(), "# total_attempted=%lld total_valid=%lld elapsed_seconds=%lld",
                        &attempted, &valid, &elapsed);
            totals = {attempted, valid, elapsed};
            continue;
        }
        if (line.empty() || line[0] == '#') continue;

        std::istringstream row(line);
        std::string cell;
        auto next = [&]() -> double {
            if (!std::getline(row, cell, ',')) throw std::invalid_argument("checkpoint row too short");
            return std::stod(cell);
        };
        int b = static_cast<int>(next());
        if (b < 0 || b >= NUM_BINS) continue;
        next(); next();  // K_lo, K_hi: unused on load
        long long count = static_cast<long long>(next());

        for (int k = 0; k < N_COMBOS; ++k) {
            BinStats& s = bins[b][k];
            s.min = next(); s.max = next(); s.mean = next(); s.nmax = next(); s.nmean = next();
            Args& w = s.worst_args;
            w.ar_h = next(); w.ar_l = next(); w.ai_h = next(); w.ai_l = next();
            w.br_h = next(); w.br_l = next(); w.bi_h = next(); w.bi_l = next();
            s.count = count;
        }
    }
    return totals;
}

inline void write_checkpoint(const Bins& bins, long long total_attempted, long long total_valid,
                              long long elapsed_seconds)
{
    static const char* ARG_NAMES[ARG_COLS] = {"ar_h", "ar_l", "ai_h", "ai_l", "br_h", "br_l", "bi_h", "bi_l"};

    std::ofstream out(RESULTS_FILE);
    out << "# Binned complex DW multiplication conditioning results (accumulated across possibly multiple runs)\n";
    out << "# combos: 0=" << COMBO_NAMES[0] << " 1=" << COMBO_NAMES[1] << " 2=" << COMBO_NAMES[2]
        << " 3=" << COMBO_NAMES[3] << " 4=" << COMBO_NAMES[4] << " 5=" << COMBO_NAMES[5] << "\n";
    out << "# total_attempted=" << total_attempted << " total_valid=" << total_valid
        << " elapsed_seconds=" << elapsed_seconds << "\n";
    out << "# nmax/nmean = max/mean of (relative error / (K * u^2)), u^2 = 2^-106, K = conditioning of the DW inputs\n";
    out << "# columns: bin,K_lo,K_hi,count";
    for (int k = 0; k < N_COMBOS; ++k) {
        out << ",c" << k << "_min,c" << k << "_max,c" << k << "_mean,c" << k << "_nmax,c" << k << "_nmean";
        for (const char* name : ARG_NAMES) out << ",c" << k << "_worst_" << name;
    }
    out << "\n";

    out << std::scientific << std::setprecision(MPFR_DISPLAY_PREC);
    for (int b = 0; b < NUM_BINS; ++b) {
        out << b << "," << std::pow(10.0, b + 1) << "," << std::pow(10.0, b + 2) << "," << bins[b][0].count;
        for (int k = 0; k < N_COMBOS; ++k) {
            const BinStats& s = bins[b][k];
            const Args& w = s.worst_args;
            out << "," << s.min << "," << s.max << "," << s.mean << "," << s.nmax << "," << s.nmean
                << "," << w.ar_h << "," << w.ar_l << "," << w.ai_h << "," << w.ai_l
                << "," << w.br_h << "," << w.br_l << "," << w.bi_h << "," << w.bi_l;
        }
        out << "\n";
    }
    out.close();

    std::ofstream log(LOG_FILE, std::ios::app);
    auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    char timestamp[32];
    std::strftime(timestamp, sizeof(timestamp), "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&now));
    log << timestamp << "  attempted=" << total_attempted << "  valid=" << total_valid
        << "  elapsed=" << elapsed_seconds << "s\n";
}

#endif  //BINNED_RESULTS_H
