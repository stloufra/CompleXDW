// Examines generate_abcd_mp: which branch produces each pair, how many attempts it takes, and how
// well the achieved K fill [K_MIN, K_MAX]. Target K is drawn exactly as in
// test_complex_dw_conditioning_binned (log-uniform), so this describes the long run's inputs.
//
// Usage: test_conditioning_generator [samples, default 100000]
// Writes res_cond/generator_samples.csv; plot with plot_conditioning_generator.py.

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>

#include <mpfr.h>

#include "../ComplexDouble.h"
#include "src/test_func.h"
#include "src/binned_ops.h"

constexpr int MAX_TRIES = 100000;  // same as the binned run

int main(int argc, char** argv)
{
    const long long samples = argc > 1 ? std::atoll(argv[1]) : 100000;
    if (samples <= 0) {
        std::cerr << "Usage: " << argv[0] << " [samples > 0]\n";
        return 1;
    }

    mpfr_set_default_prec(MPFR_PREC);
    std::mt19937_64 rng(std::random_device{}());
    std::uniform_real_distribution<double> log_k_dist(std::log10(K_MIN), std::log10(K_MAX));

    std::filesystem::create_directories("res_cond");
    std::ofstream out("res_cond/generator_samples.csv");
    out << "# generate_abcd_mp examination, target K log-uniform in [" << K_MIN << ", " << K_MAX
        << "], max_tries=" << MAX_TRIES << "\n";
    out << "# K_mpfr = K of the generated 1024-bit numbers, K_dw = K of their double-word roundings (nan if not generated)\n";
    out << "target_K,K_mpfr,K_dw,branch,attempts,tried0,tried1,tried2,tried3\n";
    out << std::scientific << std::setprecision(MPFR_DISPLAY_PREC);

    mpfr_t a, b, c, d, K, K_check;
    mpfr_inits2(MPFR_PREC, a, b, c, d, K, K_check, (mpfr_ptr) nullptr);

    auto start = std::chrono::steady_clock::now();
    long long failed = 0;
    for (long long n = 0; n < samples; ++n) {
        double target_K = std::pow(10.0, log_k_dist(rng));
        mpfr_set_d(K, target_K, MPFR_RNDN);

        GenStats st;
        double K_mpfr = NAN, K_dw = NAN;
        if (generate_abcd_mp(K, a, b, c, d, K_check, rng, MAX_TRIES, &st)) {
            K_mpfr = mpfr_get_d(K_check, MPFR_RNDN);
            K_dw = MulOp::conditioning(MulOp::make_args(a, b, c, d));
        } else {
            ++failed;
        }

        out << target_K << "," << K_mpfr << "," << K_dw << "," << st.branch << "," << st.attempts
            << "," << st.tried[0] << "," << st.tried[1] << "," << st.tried[2] << "," << st.tried[3] << "\n";

        if ((n + 1) % 100000 == 0) std::cout << "Progress: " << n + 1 << "/" << samples << "\n";
    }

    double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::cout << samples << " samples (" << failed << " failed after " << MAX_TRIES << " attempts) in "
              << std::fixed << std::setprecision(1) << secs << " s -> res_cond/generator_samples.csv\n";

    mpfr_clears(a, b, c, d, K, K_check, (mpfr_ptr) nullptr);
    mpfr_free_cache();
    return 0;
}
