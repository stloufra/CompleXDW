#ifndef BINNED_RESULTS_H
#define BINNED_RESULTS_H

// Checkpoint format shared by the binned runs (write/resume it) and the replay tools (read a stored
// worst case back). Generic over the operation under test, see binned_ops.h.

#include <array>
#include <chrono>
#include <cmath>
#include <concepts>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>

#include "test_func.h"

constexpr double K_MIN = 10.0;
constexpr double K_MAX = 1e30;
constexpr int NUM_BINS = 30;  // one per decade of K, [1e1,1e2), [1e2,1e3), ..., [1e30,1e31)
constexpr int ARG_COLS = 8;   // ar_h, ar_l, ai_h, ai_l, br_h, br_l, bi_h, bi_l
constexpr double U2 = 0x1p-106;  // u^2 for double, u = 2^-53

// The DW inputs (a, b) behind one particular result, so a worst case can be reproduced later
// with the replay tool.
struct Args {
    double ar_h = 0, ar_l = 0, ai_h = 0, ai_l = 0;
    double br_h = 0, br_l = 0, bi_h = 0, bi_l = 0;
};

// An operation a (op) b under test: its combos, the conditioning K the error is normalized by, the
// relative error of every combo, how generate_abcd_mp's (a, b, c, d) map to the DW inputs, and
// where its checkpoint lives.
template <class Op>
concept BinnedOp = requires(const Args& x, mpfr_ptr v) {
    { Op::N_COMBOS } -> std::convertible_to<int>;
    { Op::COMBO_NAMES[0] } -> std::convertible_to<const char*>;
    { Op::NAME } -> std::convertible_to<const char*>;
    { Op::K_DEFINITION } -> std::convertible_to<const char*>;
    { Op::RESULTS_FILE } -> std::convertible_to<const char*>;
    { Op::LOG_FILE } -> std::convertible_to<const char*>;
    { Op::make_args(v, v, v, v) } -> std::same_as<Args>;
    { Op::conditioning(x) } -> std::same_as<double>;
    { Op::errors(x) } -> std::same_as<std::array<double, Op::N_COMBOS>>;
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

// Exact at MPFR_PREC.
inline void args_to_mpfr(const Args& x, mpfr_t ar, mpfr_t ai, mpfr_t br, mpfr_t bi)
{
    mpfr_set_d(ar, x.ar_h, MPFR_RNDN); mpfr_add_d(ar, ar, x.ar_l, MPFR_RNDN);
    mpfr_set_d(ai, x.ai_h, MPFR_RNDN); mpfr_add_d(ai, ai, x.ai_l, MPFR_RNDN);
    mpfr_set_d(br, x.br_h, MPFR_RNDN); mpfr_add_d(br, br, x.br_l, MPFR_RNDN);
    mpfr_set_d(bi, x.bi_h, MPFR_RNDN); mpfr_add_d(bi, bi, x.bi_l, MPFR_RNDN);
}

// (|p| + |q|) / |p + q|, infinite if p + q is exactly 0.
inline double sum_conditioning(mpfr_srcptr p, mpfr_srcptr q)
{
    mpfr_t abs_p, abs_q, abs_sum;
    mpfr_inits2(MPFR_PREC, abs_p, abs_q, abs_sum, (mpfr_ptr) nullptr);
    mpfr_add(abs_sum, p, q, MPFR_RNDN);
    mpfr_abs(abs_sum, abs_sum, MPFR_RNDN);
    mpfr_abs(abs_p, p, MPFR_RNDN);
    mpfr_abs(abs_q, q, MPFR_RNDN);
    mpfr_add(abs_p, abs_p, abs_q, MPFR_RNDN);
    mpfr_div(abs_p, abs_p, abs_sum, MPFR_RNDN);
    double K = mpfr_get_d(abs_p, MPFR_RNDN);
    mpfr_clears(abs_p, abs_q, abs_sum, (mpfr_ptr) nullptr);
    return K;
}

template <BinnedOp Op>
using Bins = std::array<std::array<BinStats, Op::N_COMBOS>, NUM_BINS>;

inline int bin_index(double K)
{
    int b = static_cast<int>(std::floor(std::log10(K))) - 1;
    if (b < 0) b = 0;
    if (b >= NUM_BINS) b = NUM_BINS - 1;
    return b;
}

// Returns {total_attempted, total_valid, elapsed_seconds} from a prior checkpoint, or zeros if
// Op::RESULTS_FILE doesn't exist. Fills `bins` with what it recovers. Throws (std::invalid_argument /
// std::out_of_range) on a malformed row, e.g. a checkpoint from before the worst-case args columns.
template <BinnedOp Op>
std::array<long long, 3> load_checkpoint(Bins<Op>& bins)
{
    std::array<long long, 3> totals = {0, 0, 0};
    std::ifstream in(Op::RESULTS_FILE);
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

        for (int k = 0; k < Op::N_COMBOS; ++k) {
            BinStats& s = bins[b][k];
            s.min = next(); s.max = next(); s.mean = next(); s.nmax = next(); s.nmean = next();
            Args& w = s.worst_args;
            w.ar_h = next(); w.ar_l = next(); w.ai_h = next(); w.ai_l = next();
            w.br_h = next(); w.br_l = next(); w.bi_h = next(); w.bi_l = next();
            s.count = count;
        }
        if (std::getline(row, cell, ',')) throw std::invalid_argument("checkpoint row too long");
    }
    return totals;
}

template <BinnedOp Op>
void write_checkpoint(const Bins<Op>& bins, long long total_attempted, long long total_valid,
                      long long elapsed_seconds)
{
    static const char* ARG_NAMES[ARG_COLS] = {"ar_h", "ar_l", "ai_h", "ai_l", "br_h", "br_l", "bi_h", "bi_l"};

    std::ofstream out(Op::RESULTS_FILE);
    out << "# Binned complex DW " << Op::NAME << " conditioning results (accumulated across possibly multiple runs)\n";
    out << "# combos:";
    for (int k = 0; k < Op::N_COMBOS; ++k) out << " " << k << "=" << Op::COMBO_NAMES[k];
    out << "\n";
    out << "# total_attempted=" << total_attempted << " total_valid=" << total_valid
        << " elapsed_seconds=" << elapsed_seconds << "\n";
    out << "# K = " << Op::K_DEFINITION << ", evaluated on the DW inputs\n";
    out << "# nmax/nmean = max/mean of (relative error / (K * u^2)), u^2 = 2^-106\n";
    out << "# columns: bin,K_lo,K_hi,count";
    for (int k = 0; k < Op::N_COMBOS; ++k) {
        out << ",c" << k << "_min,c" << k << "_max,c" << k << "_mean,c" << k << "_nmax,c" << k << "_nmean";
        for (const char* name : ARG_NAMES) out << ",c" << k << "_worst_" << name;
    }
    out << "\n";

    out << std::scientific << std::setprecision(MPFR_DISPLAY_PREC);
    for (int b = 0; b < NUM_BINS; ++b) {
        out << b << "," << std::pow(10.0, b + 1) << "," << std::pow(10.0, b + 2) << "," << bins[b][0].count;
        for (int k = 0; k < Op::N_COMBOS; ++k) {
            const BinStats& s = bins[b][k];
            const Args& w = s.worst_args;
            out << "," << s.min << "," << s.max << "," << s.mean << "," << s.nmax << "," << s.nmean
                << "," << w.ar_h << "," << w.ar_l << "," << w.ai_h << "," << w.ai_l
                << "," << w.br_h << "," << w.br_l << "," << w.bi_h << "," << w.bi_l;
        }
        out << "\n";
    }
    out.close();

    std::ofstream log(Op::LOG_FILE, std::ios::app);
    auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    char timestamp[32];
    std::strftime(timestamp, sizeof(timestamp), "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&now));
    log << timestamp << "  attempted=" << total_attempted << "  valid=" << total_valid
        << "  elapsed=" << elapsed_seconds << "s\n";
}

#endif  //BINNED_RESULTS_H
