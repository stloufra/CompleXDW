// Throughput of element-wise c[i] = a[i] * b[i] and c[i] = a[i] / b[i] for every mul<Add,Norm> and
// div<Div,Add,Norm> variant, against plain std::complex and naive (no inf/nan check) baselines, for double and float.
//
// Each (variant, length) is timed SAMPLES times; a sample repeats the kernel for at least
// MIN_SAMPLE_TIME so the clock resolution doesn't affect it and the variant order rotates between
// samples so frequency/thermal drift spreads evenly. Writes results/elementwise_<BENCH_BUILD>.csv.

#include <algorithm>
#include <array>
#include <chrono>
#include <complex>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#ifdef __APPLE__
#include <pthread/qos.h>
#endif

#include "ComplexDouble.h"
#include "random_dw.h"

#ifndef BENCH_BUILD
#define BENCH_BUILD "default"
#endif

using namespace XDW_ARTH;

constexpr std::array<std::size_t, 7> SIZES = {256, 1024, 4096, 16384, 65536, 262144, 1048576};

// a, b, c all in L1 on M
// P-core L1D is 128 kB. Three arrays of 42 kB fit 1300 XDW -> N = 1024
constexpr std::size_t REFERENCE_SIZE = 1024;  
constexpr int SAMPLES = 31;
constexpr std::chrono::nanoseconds MIN_SAMPLE_TIME = std::chrono::milliseconds(1);

template <std::floating_point T>
struct Buffers {
    std::vector<ComplexDouble<T>> a, b, c;
    std::vector<std::complex<T>> sa, sb, sc;  // high words of a, b for the baselines

    Buffers(std::size_t n, std::mt19937_64& rng) : a(n), b(n), c(n), sa(n), sb(n), sc(n)
    {
        for (std::size_t i = 0; i < n; ++i) {
            a[i] = random_dw_complex<T>(rng);
            b[i] = random_dw_complex<T>(rng);
            sa[i] = {a[i].re_h(), a[i].im_h()};
            sb[i] = {b[i].re_h(), b[i].im_h()};
        }
    }
};

template <std::floating_point T>
using Kernel = void (*)(Buffers<T>&, std::size_t);

template <std::floating_point T, AddMode Add, NormMode Norm>
[[gnu::noinline]] void dw_mul(Buffers<T>& buf, std::size_t n)
{
    const ComplexDouble<T>* __restrict__ a = buf.a.data();
    const ComplexDouble<T>* __restrict__ b = buf.b.data();
    ComplexDouble<T>* __restrict__ c = buf.c.data();
    for (std::size_t i = 0; i < n; ++i) c[i] = ComplexDouble<T>::template mul<Add, Norm>(a[i], b[i]);
}

template <std::floating_point T, DivMode Div, AddMode Add, NormMode Norm>
[[gnu::noinline]] void dw_div(Buffers<T>& buf, std::size_t n)
{
    const ComplexDouble<T>* __restrict__ a = buf.a.data();
    const ComplexDouble<T>* __restrict__ b = buf.b.data();
    ComplexDouble<T>* __restrict__ c = buf.c.data();
    for (std::size_t i = 0; i < n; ++i) c[i] = ComplexDouble<T>::template div<Div, Add, Norm>(a[i], b[i]);
}

// like std:: but no check undeflow/overflow so vectorizes 
template <std::floating_point T>
[[gnu::noinline]] void naive_mul(Buffers<T>& buf, std::size_t n)
{
    const std::complex<T>* __restrict__ a = buf.sa.data();
    const std::complex<T>* __restrict__ b = buf.sb.data();
    std::complex<T>* __restrict__ c = buf.sc.data();
    for (std::size_t i = 0; i < n; ++i) {
        T ar = a[i].real(), ai = a[i].imag(), br = b[i].real(), bi = b[i].imag();
        c[i] = {ar * br - ai * bi, ar * bi + ai * br};
    }
}

template <std::floating_point T>
[[gnu::noinline]] void naive_div(Buffers<T>& buf, std::size_t n)
{
    const std::complex<T>* __restrict__ a = buf.sa.data();
    const std::complex<T>* __restrict__ b = buf.sb.data();
    std::complex<T>* __restrict__ c = buf.sc.data();
    for (std::size_t i = 0; i < n; ++i) {
        T ar = a[i].real(), ai = a[i].imag(), br = b[i].real(), bi = b[i].imag();
        T denom = br * br + bi * bi;
        c[i] = {(ar * br + ai * bi) / denom, (ai * br - ar * bi) / denom};
    }
}

template <std::floating_point T>
[[gnu::noinline]] void std_mul(Buffers<T>& buf, std::size_t n)
{
    const std::complex<T>* __restrict__ a = buf.sa.data();
    const std::complex<T>* __restrict__ b = buf.sb.data();
    std::complex<T>* __restrict__ c = buf.sc.data();
    for (std::size_t i = 0; i < n; ++i) c[i] = a[i] * b[i];
}

template <std::floating_point T>
[[gnu::noinline]] void std_div(Buffers<T>& buf, std::size_t n)
{
    const std::complex<T>* __restrict__ a = buf.sa.data();
    const std::complex<T>* __restrict__ b = buf.sb.data();
    std::complex<T>* __restrict__ c = buf.sc.data();
    for (std::size_t i = 0; i < n; ++i) c[i] = a[i] / b[i];
}

constexpr const char* name(AddMode m)
{
    return m == AddMode::Madd ? "Madd" : m == AddMode::Accurate ? "Accurate" : "Sloppy";
}
constexpr const char* name(NormMode m) { return m == NormMode::Normalized ? "Normalized" : "Unnormalized"; }
constexpr const char* name(DivMode m) { return m == DivMode::Div2 ? "Div2" : "Div3"; }

// two DWTimesDW2 2*(8 norm, 5 unnorm) 
// one DWadd (20 Madd/Accurate, 11 Sloppy).
constexpr int dw_add_flops(AddMode add) { return add == AddMode::Sloppy ? 11 : 20; }
constexpr int mul_add_flops(AddMode add, NormMode norm) { return (norm == NormMode::Normalized ? 16 : 10) + dw_add_flops(add); }

// DENOM
// two DWPow2 2*(7 norm, 4 unnorm)
// one DWadd (20 Madd/Accurate, 11 Sloppy).
// NUMER
// mul-adds
// two DWDivDW2 2*(18 each), or one DWRecip of the shared denominator (22, 1 division)
// and two DWTimesDW3 2*(9 each).
constexpr int div_flops(DivMode div, AddMode add, NormMode norm)
{
    int denom = (norm == NormMode::Normalized ? 14 : 8) + dw_add_flops(add);
    int final_div = div == DivMode::Div2 ? 2 * 18 : 22 + 2 * 9;
    return denom + 2 * mul_add_flops(add, NormMode::Normalized) + final_div;
}
constexpr int div_divisions(DivMode div) { return div == DivMode::Div2 ? 4 : 1; }

template <std::floating_point T>
struct Variant {
    const char* op;  // "mul" or "div"
    std::string name;
    int flops;       // 0 if unknown
    int divisions;
    Kernel<T> kernel;
};

template <std::floating_point T, AddMode Add, NormMode Norm>
Variant<T> mul_variant()
{
    return {"mul", std::string(name(Add)) + "/" + name(Norm), 2 * mul_add_flops(Add, Norm), 0, dw_mul<T, Add, Norm>};
}

template <std::floating_point T, DivMode Div, AddMode Add, NormMode Norm>
Variant<T> div_variant()
{
    return {"div", std::string(name(Div)) + "/" + name(Add) + "/" + name(Norm), div_flops(Div, Add, Norm),
            div_divisions(Div), dw_div<T, Div, Add, Norm>};
}

// The first DW variant of each op is the reference the others are compared to.
template <std::floating_point T>
std::vector<Variant<T>> variants()
{
    return {
        mul_variant<T, AddMode::Madd, NormMode::Normalized>(),
        mul_variant<T, AddMode::Madd, NormMode::Unnormalized>(),
        mul_variant<T, AddMode::Accurate, NormMode::Normalized>(),
        mul_variant<T, AddMode::Accurate, NormMode::Unnormalized>(),
        mul_variant<T, AddMode::Sloppy, NormMode::Normalized>(),
        mul_variant<T, AddMode::Sloppy, NormMode::Unnormalized>(),
        {"mul", "naive", 6, 0, naive_mul<T>},
        {"mul", "std::complex", 0, 0, std_mul<T>},

        div_variant<T, DivMode::Div2, AddMode::Madd, NormMode::Normalized>(),
        div_variant<T, DivMode::Div2, AddMode::Madd, NormMode::Unnormalized>(),
        div_variant<T, DivMode::Div2, AddMode::Accurate, NormMode::Normalized>(),
        div_variant<T, DivMode::Div2, AddMode::Accurate, NormMode::Unnormalized>(),
        div_variant<T, DivMode::Div2, AddMode::Sloppy, NormMode::Normalized>(),
        div_variant<T, DivMode::Div2, AddMode::Sloppy, NormMode::Unnormalized>(),
        div_variant<T, DivMode::Div3, AddMode::Madd, NormMode::Normalized>(),
        div_variant<T, DivMode::Div3, AddMode::Madd, NormMode::Unnormalized>(),
        div_variant<T, DivMode::Div3, AddMode::Accurate, NormMode::Normalized>(),
        div_variant<T, DivMode::Div3, AddMode::Accurate, NormMode::Unnormalized>(),
        div_variant<T, DivMode::Div3, AddMode::Sloppy, NormMode::Normalized>(),
        div_variant<T, DivMode::Div3, AddMode::Sloppy, NormMode::Unnormalized>(),
        {"div", "naive", 11, 2, naive_div<T>},
        {"div", "std::complex", 0, 0, std_div<T>},
    };
}

template <std::floating_point T>
double time_reps(const Variant<T>& v, Buffers<T>& buf, std::size_t n, long reps)
{
    auto start = std::chrono::steady_clock::now();
    for (long r = 0; r < reps; ++r) {
        v.kernel(buf, n);
        asm volatile("" ::: "memory");
    }
    return std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - start).count();
}

// Also serves as the warm-up.
template <std::floating_point T>
long calibrate_reps(const Variant<T>& v, Buffers<T>& buf, std::size_t n)
{
    long reps = 1;
    while (time_reps(v, buf, n, reps) < MIN_SAMPLE_TIME.count()) reps *= 2;
    return reps;
}

double median(std::vector<double> x)
{
    std::sort(x.begin(), x.end());
    return x[x.size() / 2];
}

template <std::floating_point T>
void print_summary(const char* type, const std::vector<Variant<T>>& vs, const std::vector<std::vector<double>>& ns)
{
    std::cout << "\n" << type << ", n = " << REFERENCE_SIZE << " (" << BENCH_BUILD << " build), ns/element:\n";
    std::cout << std::fixed;
    const Variant<T>* ref = nullptr;
    double ref_median = 0;
    for (std::size_t k = 0; k < vs.size(); ++k) {
        const Variant<T>& v = vs[k];
        double m = median(ns[k]);
        if (!ref || ref->op != std::string(v.op)) { ref = &v; ref_median = m; }
        std::cout << "  " << v.op << "  " << std::left << std::setw(28) << v.name << std::right
                  << std::setprecision(3) << std::setw(8) << m << "  min " << std::setw(8)
                  << *std::min_element(ns[k].begin(), ns[k].end()) << "   x" << std::setprecision(2)
                  << m / ref_median;
        if (v.flops > 0) std::cout << "  (flops x" << static_cast<double>(v.flops) / ref->flops << ")";
        std::cout << "\n";
    }
}

template <std::floating_point T>
void run(const char* type, std::ofstream& csv, std::mt19937_64& rng)
{
    const std::vector<Variant<T>> vs = variants<T>();
    for (std::size_t n : SIZES) {
        Buffers<T> buf(n, rng);
        std::vector<long> reps(vs.size());
        for (std::size_t k = 0; k < vs.size(); ++k) reps[k] = calibrate_reps(vs[k], buf, n);

        std::vector<std::vector<double>> ns(vs.size());
        for (int s = 0; s < SAMPLES; ++s) {
            for (std::size_t j = 0; j < vs.size(); ++j) {
                std::size_t k = (j + s) % vs.size();
                vs[k].kernel(buf, n);
                double t = time_reps(vs[k], buf, n, reps[k]) / (static_cast<double>(reps[k]) * n);
                ns[k].push_back(t);
                csv << BENCH_BUILD << "," << type << "," << vs[k].op << "," << vs[k].name << "," << vs[k].flops
                    << "," << vs[k].divisions << "," << n << "," << s << "," << t << "\n";
            }
        }
        std::cerr << type << " n=" << n << " done\n";
        if (n == REFERENCE_SIZE) print_summary(type, vs, ns);
    }
}

int main()
{
#ifdef __APPLE__
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);  // prefer a performance core
#endif
    std::mt19937_64 rng(std::random_device{}());

    const std::string path = std::string("results/elementwise_") + BENCH_BUILD + ".csv";
    std::ofstream csv(path);
    if (!csv) {
        std::cerr << "Cannot write " << path << " (run from perf/)\n";
        return 1;
    }
    csv << "build,type,op,variant,flops,divisions,n,sample,ns_per_elem\n";
    csv << std::setprecision(6);

    run<double>("double", csv, rng);
    run<float>("float", csv, rng);
    std::cout << "\nWrote " << path << "\n";
    return 0;
}
