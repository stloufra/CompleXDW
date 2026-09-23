// Long-running complex DW multiplication conditioning sweep, meant for a multi-day cluster job.
//
// Unlike test_complex_dw_conditioning.cpp (one fixed-size deterministic K sweep, all raw samples
// kept), this runs indefinitely over randomly-conditioned samples and keeps only aggregated
// per-decade-of-K statistics (min/max/mean per AddMode/NormMode combo), so memory and output size
// stay flat no matter how long it runs. It checkpoints periodically (res/binned_results.csv,
// overwritten with the current aggregate; res/binned_log.txt, appended to) and resumes from an
// existing checkpoint on startup, so a job can be killed/resubmitted without losing progress.
//
// Stop with Ctrl-C or SIGTERM (e.g. a cluster job hitting its time limit): both trigger one final
// checkpoint before exit.

#include <array>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <unistd.h>

#include <mpfr.h>

#include "../ComplexDouble.h"
#include "src/test_func.h"
#include "src/combo_mul.h"
#include "src/binned_results.h"

using namespace XDW_ARTH;

constexpr int CHECKPOINT_INTERVAL_SECONDS = 60;

static volatile sig_atomic_t g_stop = 0;
static void handle_stop_signal(int) { g_stop = 1; }

// Wall-clock time alone repeats across parallel cluster launches started in the same instant,
// which would make every instance draw the identical "random" K/a/b/c/d sequence. Mix in
// random_device, PID, and (when present) a cluster scheduler's per-task ID to keep launches
// independent even when started simultaneously.
static std::mt19937_64 make_rng()
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

int main()
{
    std::signal(SIGINT, handle_stop_signal);
    std::signal(SIGTERM, handle_stop_signal);

    mpfr_set_default_prec(MPFR_PREC);
    std::mt19937_64 rng = make_rng();
    std::uniform_real_distribution<double> log_k_dist(std::log10(K_MIN), std::log10(K_MAX));

    Bins bins{};
    std::array<long long, 3> totals;
    try {
        totals = load_checkpoint(bins);
    } catch (const std::exception& e) {
        std::cerr << "Cannot resume from " << RESULTS_FILE << " (" << e.what() << "). "
                  << "It is probably from an older format: move or delete it to start fresh.\n";
        return 1;
    }
    long long total_attempted = totals[0], total_valid = totals[1], resumed_elapsed_seconds = totals[2];
    if (total_attempted > 0) {
        std::cout << "Resumed from " << RESULTS_FILE << ": " << total_attempted << " attempted, "
                  << total_valid << " valid, " << resumed_elapsed_seconds << "s elapsed so far.\n";
    }

    mpfr_t ar, ai, br, bi, K_mpfr, K_check;
    mpfr_init2(ar, MPFR_PREC); mpfr_init2(ai, MPFR_PREC);
    mpfr_init2(br, MPFR_PREC); mpfr_init2(bi, MPFR_PREC);
    mpfr_init2(K_mpfr, MPFR_PREC); mpfr_init2(K_check, MPFR_PREC);

    auto run_start = std::chrono::steady_clock::now();
    auto last_checkpoint = run_start;

    while (!g_stop) {
        double target_K = std::pow(10.0, log_k_dist(rng));
        ++total_attempted;

        mpfr_set_d(K_mpfr, target_K, MPFR_RNDN);
        if (generate_abcd_mp(K_mpfr, ar, ai, br, bi, K_check, rng, 100000)) {
            double ar_h, ar_l, ai_h, ai_l, br_h, br_l, bi_h, bi_l;
            mpfr_to_dw(ar, MPFR_RNDN, &ar_h, &ar_l);
            mpfr_to_dw(ai, MPFR_RNDN, &ai_h, &ai_l);
            mpfr_to_dw(br, MPFR_RNDN, &br_h, &br_l);
            mpfr_to_dw(bi, MPFR_RNDN, &bi_h, &bi_l);

            // K and the errors are both taken on the DW inputs, so bin/normalization match a replay.
            Args args{ar_h, ar_l, ai_h, ai_l, br_h, br_l, bi_h, bi_l};
            double K = conditioning(args);
            if (std::isfinite(K) && K > 0) {
                int bin = bin_index(K);
                std::array<double, N_COMBOS> err = combo_errors(args);
                for (int k = 0; k < N_COMBOS; ++k) bins[bin][k].update(err[k], err[k] / (K * U2), args);
                ++total_valid;
            }
        }

        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - last_checkpoint).count() >= CHECKPOINT_INTERVAL_SECONDS) {
            long long elapsed = resumed_elapsed_seconds
                + std::chrono::duration_cast<std::chrono::seconds>(now - run_start).count();
            write_checkpoint(bins, total_attempted, total_valid, elapsed);
            last_checkpoint = now;
        }
    }

    long long elapsed = resumed_elapsed_seconds
        + std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - run_start).count();
    write_checkpoint(bins, total_attempted, total_valid, elapsed);
    std::cout << "\nStopped. " << total_attempted << " attempted, " << total_valid << " valid, "
              << elapsed << "s total elapsed. Checkpoint written to " << RESULTS_FILE << "\n";

    mpfr_clear(ar); mpfr_clear(ai); mpfr_clear(br); mpfr_clear(bi);
    mpfr_clear(K_mpfr); mpfr_clear(K_check);
    mpfr_free_cache();
    return 0;
}
