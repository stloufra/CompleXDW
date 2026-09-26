#ifndef BINNED_RUN_H
#define BINNED_RUN_H

// The long-running binned conditioning sweep and its worst-case replay, generic over the operation.
//
// The sweep runs indefinitely over randomly-conditioned samples and keeps only aggregated
// per-decade-of-K statistics per combo, so memory and output size stay flat no matter how long it
// runs. It checkpoints every CHECKPOINT_INTERVAL_SECONDS (Op::RESULTS_FILE overwritten with the
// current aggregate, Op::LOG_FILE appended to) and resumes from an existing checkpoint on startup.
// Ctrl-C or SIGTERM (e.g. a cluster job hitting its time limit) trigger one final checkpoint.

#include <array>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <unistd.h>

#include <mpfr.h>

#include "test_func.h"
#include "binned_results.h"

constexpr int CHECKPOINT_INTERVAL_SECONDS = 60;
constexpr int GENERATOR_MAX_TRIES = 100000;

inline volatile sig_atomic_t g_stop = 0;
inline void handle_stop_signal(int) { g_stop = 1; }

// Wall-clock time alone repeats across parallel cluster launches started in the same instant,
// which would make every instance draw the identical "random" K/a/b/c/d sequence. Mix in
// random_device, PID, and (when present) a cluster scheduler's per-task ID to keep launches
// independent even when started simultaneously.
inline std::mt19937_64 make_rng()
{
    std::random_device rd;
    auto now = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    unsigned task_id = 0;
    for (const char* var : {"SLURM_PROCID", "SLURM_ARRAY_TASK_ID", "SLURM_JOB_ID", "PBS_ARRAYID", "PBS_JOBID"}) {
        if (const char* v = std::getenv(var)) { task_id = static_cast<unsigned>(std::hash<std::string>{}(v)); break; }
    }
    std::seed_seq seed{ rd(), rd(), rd(), rd(),
                        static_cast<unsigned>(now), static_cast<unsigned>(now >> 32),
                        static_cast<unsigned>(getpid()), task_id };
    return std::mt19937_64(seed);
}

template <BinnedOp Op>
int run_binned()
{
    std::signal(SIGINT, handle_stop_signal);
    std::signal(SIGTERM, handle_stop_signal);

    mpfr_set_default_prec(MPFR_PREC);
    std::mt19937_64 rng = make_rng();
    std::uniform_real_distribution<double> log_k_dist(std::log10(K_MIN), std::log10(K_MAX));

    Bins<Op> bins{};
    std::array<long long, 3> totals;
    try {
        totals = load_checkpoint<Op>(bins);
    } catch (const std::exception& e) {
        std::cerr << "Cannot resume from " << Op::RESULTS_FILE << " (" << e.what() << "). "
                  << "It is probably from an older format or another operation: move or delete it to start fresh.\n";
        return 1;
    }
    long long total_attempted = totals[0], total_valid = totals[1], resumed_elapsed_seconds = totals[2];
    if (total_attempted > 0) {
        std::cout << "Resumed from " << Op::RESULTS_FILE << ": " << total_attempted << " attempted, "
                  << total_valid << " valid, " << resumed_elapsed_seconds << "s elapsed so far.\n";
    }

    mpfr_t a, b, c, d, K_mpfr, K_check;
    mpfr_inits2(MPFR_PREC, a, b, c, d, K_mpfr, K_check, (mpfr_ptr) nullptr);

    auto run_start = std::chrono::steady_clock::now();
    auto last_checkpoint = run_start;
    auto elapsed_since = [&](auto now) {
        return resumed_elapsed_seconds + std::chrono::duration_cast<std::chrono::seconds>(now - run_start).count();
    };

    while (!g_stop) {
        ++total_attempted;
        mpfr_set_d(K_mpfr, std::pow(10.0, log_k_dist(rng)), MPFR_RNDN);
        if (generate_abcd_mp(K_mpfr, a, b, c, d, K_check, rng, GENERATOR_MAX_TRIES)) {
            // K and the errors are both taken on the DW inputs, so bin/normalization match a replay.
            const Args args = Op::make_args(a, b, c, d);
            double K = Op::conditioning(args);
            if (std::isfinite(K) && K > 0) {
                auto& row = bins[bin_index(K)];
                const auto err = Op::errors(args);
                for (int k = 0; k < Op::N_COMBOS; ++k) row[k].update(err[k], err[k] / (K * U2), args);
                ++total_valid;
            }
        }

        auto now = std::chrono::steady_clock::now();
        if (now - last_checkpoint >= std::chrono::seconds(CHECKPOINT_INTERVAL_SECONDS)) {
            write_checkpoint<Op>(bins, total_attempted, total_valid, elapsed_since(now));
            last_checkpoint = now;
        }
    }

    long long elapsed = elapsed_since(std::chrono::steady_clock::now());
    write_checkpoint<Op>(bins, total_attempted, total_valid, elapsed);
    std::cout << "\nStopped. " << total_attempted << " attempted, " << total_valid << " valid, "
              << elapsed << "s total elapsed. Checkpoint written to " << Op::RESULTS_FILE << "\n";

    mpfr_clears(a, b, c, d, K_mpfr, K_check, (mpfr_ptr) nullptr);
    mpfr_free_cache();
    return 0;
}

// Usage: <prog> <bin> <combo>. Prints the stored worst-case inputs, their K and the error of every
// combo on them. Returns 0 if the selected combo reproduces the stored maximum exactly, 2 if not.
template <BinnedOp Op>
int replay_binned(int argc, char** argv)
{
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <bin 0.." << NUM_BINS - 1 << "> <combo 0.." << Op::N_COMBOS - 1 << ">\n";
        return 1;
    }
    int bin = std::atoi(argv[1]);
    int combo = std::atoi(argv[2]);
    if (bin < 0 || bin >= NUM_BINS || combo < 0 || combo >= Op::N_COMBOS) {
        std::cerr << "bin or combo out of range\n";
        return 1;
    }

    mpfr_set_default_prec(MPFR_PREC);

    Bins<Op> bins{};
    load_checkpoint<Op>(bins);
    const BinStats& stored = bins[bin][combo];
    if (stored.count == 0) {
        std::cerr << "No samples stored for bin " << bin << " in " << Op::RESULTS_FILE << "\n";
        return 1;
    }
    const Args& x = stored.worst_args;

    std::cout << std::scientific << std::setprecision(MPFR_DISPLAY_PREC);
    std::cout << Op::NAME << " a " << Op::SYMBOL << " b, bin " << bin << " (K in [1e" << bin + 1 << ", 1e"
              << bin + 2 << ")), combo " << combo << " = " << Op::COMBO_NAMES[combo] << ", "
              << stored.count << " samples in bin\n";
    std::cout << "a = (" << x.ar_h << ", " << x.ar_l << ") + i(" << x.ai_h << ", " << x.ai_l << ")\n";
    std::cout << "b = (" << x.br_h << ", " << x.br_l << ") + i(" << x.bi_h << ", " << x.bi_l << ")\n";
    std::cout << "K = " << Op::conditioning(x) << "\n\n";

    const auto err = Op::errors(x);
    for (int k = 0; k < Op::N_COMBOS; ++k)
        std::cout << (k == combo ? "* " : "  ") << Op::COMBO_NAMES[k] << ": " << err[k] << "\n";

    bool reproduced = err[combo] == stored.max;
    std::cout << "\nstored max = " << stored.max << ", recomputed = " << err[combo]
              << (reproduced ? "  -> reproduced\n" : "  -> DIFFERS\n");

    mpfr_free_cache();
    return reproduced ? 0 : 2;
}

#endif  //BINNED_RUN_H
