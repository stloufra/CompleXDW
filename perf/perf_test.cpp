#include <iostream>
#include <iomanip>
#include <array>
#include <memory>
#include <random>
#include <chrono>
#include <cmath>
#include <fstream>
#include <omp.h>

#include "../ComplexDouble.h"
#include "../test/src/test_func.h"

constexpr int K = 100;

using namespace XDW_ARTH;

template<size_t N>
void save_results(std::array<double, K>& times_acc_norm,
                 std::array<double, K>& times_acc_un,
                 std::array<double, K>& times_sloppy_un) {
    std::ofstream out("results/perf_times.csv", std::ios::app);
    for (int k = 0; k < K; ++k) {
        out << N << "," << k << "," << times_acc_norm[k] << "," 
            << times_acc_un[k] << "," << times_sloppy_un[k] << "\n";
    }
    out.close();
}

template<size_t N>
double measure_time_impl(ComplexDouble<double>* a, 
                        ComplexDouble<double>* b,
                        int warmup, int iterations) {
    alignas(64) std::array<double, N> results;
    
    for (int i = 0; i < warmup; ++i) {
        #pragma omp simd
        for (size_t i = 0; i < N; ++i) {
            auto c = mul_accurate_norm(a[i], b[i]);
            results[i] = c.re_h();
        }
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < iterations; ++iter) {
        #pragma omp simd
        for (size_t i = 0; i < N; ++i) {
            auto c = mul_accurate_norm(a[i], b[i]);
            results[i] = c.re_h();
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    
    volatile double sink = results[0];
    // inline asm no-op: read+write sink to register, prevents dead-store elim
    asm volatile("" : "+r"(sink));
    
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    return static_cast<double>(duration.count()) / iterations;
}

template<size_t N>
double measure_time_impl_un(ComplexDouble<double>* a, 
                           ComplexDouble<double>* b,
                           int warmup, int iterations) {
    alignas(64) std::array<double, N> results;
    
    for (int i = 0; i < warmup; ++i) {
        #pragma omp simd
        for (size_t i = 0; i < N; ++i) {
            auto c = mul_accurate_unnorm(a[i], b[i]);
            results[i] = c.re_h();
        }
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < iterations; ++iter) {
        #pragma omp simd
        for (size_t i = 0; i < N; ++i) {
            auto c = mul_accurate_unnorm(a[i], b[i]);
            results[i] = c.re_h();
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    
    volatile double sink = results[0];
    // inline asm no-op: read+write sink to register, prevents dead-store elim
    asm volatile("" : "+r"(sink));
    
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    return static_cast<double>(duration.count()) / iterations;
}

template<size_t N>
double measure_time_impl_sloppy(ComplexDouble<double>* a, 
                               ComplexDouble<double>* b,
                               int warmup, int iterations) {
    alignas(64) std::array<double, N> results;
    
    for (int i = 0; i < warmup; ++i) {
        #pragma omp simd
        for (size_t i = 0; i < N; ++i) {
            auto c = mul_sloppy_unnorm(a[i], b[i]);
            results[i] = c.re_h();
        }
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < iterations; ++iter) {
        #pragma omp simd
        for (size_t i = 0; i < N; ++i) {
            auto c = mul_sloppy_unnorm(a[i], b[i]);
            results[i] = c.re_h();
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    
    volatile double sink = results[0];
    // inline asm no-op: read+write sink to register, prevents dead-store elim
    asm volatile("" : "+r"(sink));
    
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    return static_cast<double>(duration.count()) / iterations;
}

template<size_t N>
void run_benchmark() {
    const int warmup = 10;
    const int iterations = 100;
    
    std::mt19937_64 rng(std::chrono::high_resolution_clock::now().time_since_epoch().count());
    
    auto a = std::make_unique<std::array<ComplexDouble<double>, N>>();
    auto b = std::make_unique<std::array<ComplexDouble<double>, N>>();
    
    generate_random_dw_complex<N>(*a, *b, rng);
    
    std::array<double, K> times_acc_norm;
    std::array<double, K> times_acc_un;
    std::array<double, K> times_sloppy_un;
    
    for (int k = 0; k < K; ++k) {
        times_acc_norm[k] = measure_time_impl<N>(a->data(), b->data(), warmup, iterations);
        times_acc_un[k] = measure_time_impl_un<N>(a->data(), b->data(), warmup, iterations);
        times_sloppy_un[k] = measure_time_impl_sloppy<N>(a->data(), b->data(), warmup, iterations);
    }
    
    double mean_acc_norm = 0, mean_acc_un = 0, mean_sloppy = 0;
    double var_acc_norm = 0, var_acc_un = 0, var_sloppy = 0;
    
    for (int k = 0; k < K; ++k) {
        mean_acc_norm += times_acc_norm[k];
        mean_acc_un += times_acc_un[k];
        mean_sloppy += times_sloppy_un[k];
    }
    mean_acc_norm /= K;
    mean_acc_un /= K;
    mean_sloppy /= K;
    
    for (int k = 0; k < K; ++k) {
        var_acc_norm += (times_acc_norm[k] - mean_acc_norm) * (times_acc_norm[k] - mean_acc_norm);
        var_acc_un += (times_acc_un[k] - mean_acc_un) * (times_acc_un[k] - mean_acc_un);
        var_sloppy += (times_sloppy_un[k] - mean_sloppy) * (times_sloppy_un[k] - mean_sloppy);
    }
    var_acc_norm /= K;
    var_acc_un /= K;
    var_sloppy /= K;
    
    double std_acc_norm = std::sqrt(var_acc_norm);
    double std_acc_un = std::sqrt(var_acc_un);
    double std_sloppy = std::sqrt(var_sloppy);
    
    std::cout << "Results for array_size=" << N << "\n";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  mul_accurate_norm:  " << mean_acc_norm << " +/- " << std_acc_norm 
              << " (" << 100*std_acc_norm/mean_acc_norm << "%) ns/op\n";
    std::cout << "  mul_accurate_unnorm: " << mean_acc_un << " +/- " << std_acc_un 
              << " (" << 100*std_acc_un/mean_acc_un << "%) ns/op\n";
    std::cout << "  mul_sloppy_unnorm:   " << mean_sloppy << " +/- " << std_sloppy 
              << " (" << 100*std_sloppy/mean_sloppy << "%) ns/op\n";
    std::cout << "\n";
    
    save_results<N>(times_acc_norm, times_acc_un, times_sloppy_un);
}

int main() {
    std::cout << "Performance test\n";
    std::cout << "K=" << K << " measurements per benchmark\n\n";
    
    std::ofstream out("results/perf_times.csv");
    out << "array_size,iteration,time_acc_norm,time_acc_un,time_sloppy_un\n";
    out.close();
    
    run_benchmark<50>();
    run_benchmark<100>();
    run_benchmark<1000>();
    run_benchmark<5000>();
    run_benchmark<10000>();
    run_benchmark<50000>();
    run_benchmark<100000>();
    run_benchmark<500000>();
    
    return 0;
}
