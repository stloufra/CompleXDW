#include <cstddef>
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

constexpr int MEASUREMENTS = 100;
constexpr int WARMUPS = 10;
constexpr int REAL = 100;

using namespace XDW_ARTH;

template<size_t N>
void save_results(std::array<double, MEASUREMENTS>& times_acc_norm,
                std::array<double, MEASUREMENTS>& times_acc_un,
                std::array<double, MEASUREMENTS>& times_sloppy_un) {
    std::ofstream out("results/perf_times.csv", std::ios::app);
    for (int k = 0; k < MEASUREMENTS; ++k) {
        out << N << "," << k << "," << times_acc_norm[k]/N << "," 
            << times_acc_un[k]/N << "," << times_sloppy_un[k]/N << "\n";
    }
    out.close();
}

template<size_t N, size_t ITER>
double measure_time_norm_acc(ComplexDouble<double>* a, 
                              ComplexDouble<double>* b,
                              ComplexDouble<double>* c) {
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t r = 0; r < ITER; ++r) {
        for (size_t i = 0; i < N; ++i) {
            c[i] = mul_accurate_norm(a[i], b[i]);
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    volatile double sink = c[0].re_h();
    asm volatile("" : "+r"(sink));
    return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / ITER;
}

template<size_t N, size_t ITER>
double measure_time_un_acc(ComplexDouble<double>* a, 
                           ComplexDouble<double>* b,
                           ComplexDouble<double>* c) {
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t r = 0; r < ITER; ++r) {
        for (size_t i = 0; i < N; ++i) {
            c[i] = mul_accurate_unnorm(a[i], b[i]);
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    volatile double sink = c[0].re_h();
    asm volatile("" : "+r"(sink));
    return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / ITER;
}

template<size_t N, size_t ITER>
double measure_time_un_sloppy(ComplexDouble<double>* a, 
                              ComplexDouble<double>* b,
                              ComplexDouble<double>* c) {
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t r = 0; r < ITER; ++r) {
        for (size_t i = 0; i < N; ++i) {
            c[i] = mul_sloppy_unnorm(a[i], b[i]);
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    volatile double sink = c[0].re_h();
    asm volatile("" : "+r"(sink));
    return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / ITER;
}

template<size_t N>
void run_benchmark() {
    std::mt19937_64 rng(std::chrono::high_resolution_clock::now().time_since_epoch().count());
    
    auto a = std::make_unique<std::array<ComplexDouble<double>, N>>();
    auto b = std::make_unique<std::array<ComplexDouble<double>, N>>();
    auto c = std::make_unique<std::array<ComplexDouble<double>, N>>();
    
    generate_random_dw_complex<N>(*a, *b, rng);
    
    std::array<double, MEASUREMENTS> times_acc_norm;
    std::array<double, MEASUREMENTS> times_acc_un;
    std::array<double, MEASUREMENTS> times_sloppy_un;
    
    for (int m = 0; m < MEASUREMENTS; ++m) {
        measure_time_norm_acc<N, WARMUPS>(a->data(), b->data(), c->data());
        measure_time_un_acc<N, WARMUPS>(a->data(), b->data(), c->data());
        measure_time_un_sloppy<N, WARMUPS>(a->data(), b->data(), c->data());
        
        times_acc_norm[m] = measure_time_norm_acc<N, REAL>(a->data(), b->data(), c->data());
        times_acc_un[m] = measure_time_un_acc<N, REAL>(a->data(), b->data(), c->data());
        times_sloppy_un[m] = measure_time_un_sloppy<N, REAL>(a->data(), b->data(), c->data());
    }
    
    double mean_acc_norm = 0, mean_acc_un = 0, mean_sloppy = 0;
    double var_acc_norm = 0, var_acc_un = 0, var_sloppy = 0;
    
    for (int k = 0; k < MEASUREMENTS; ++k) {
        mean_acc_norm += times_acc_norm[k];
        mean_acc_un += times_acc_un[k];
        mean_sloppy += times_sloppy_un[k];
    }
    mean_acc_norm /= MEASUREMENTS;
    mean_acc_un /= MEASUREMENTS;
    mean_sloppy /= MEASUREMENTS;
    
    for (int k = 0; k < MEASUREMENTS; ++k) {
        var_acc_norm += (times_acc_norm[k] - mean_acc_norm) * (times_acc_norm[k] - mean_acc_norm);
        var_acc_un += (times_acc_un[k] - mean_acc_un) * (times_acc_un[k] - mean_acc_un);
        var_sloppy += (times_sloppy_un[k] - mean_sloppy) * (times_sloppy_un[k] - mean_sloppy);
    }
    var_acc_norm /= MEASUREMENTS;
    var_acc_un /= MEASUREMENTS;
    var_sloppy /= MEASUREMENTS;
    
    double std_acc_norm = std::sqrt(var_acc_norm);
    double std_acc_un = std::sqrt(var_acc_un);
    double std_sloppy = std::sqrt(var_sloppy);
    
    std::cout << "Results for array_size=" << N << "\n";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  mul_accurate_norm:  " << mean_acc_norm/N << " +/- " << std_acc_norm/N 
              << " (" << 100*std_acc_norm/mean_acc_norm << "%) ns/op\n";
    std::cout << "  mul_accurate_unnorm: " << mean_acc_un/N << " +/- " << std_acc_un/N 
              << " (" << 100*std_acc_un/mean_acc_un << "%) ns/op\n";
    std::cout << "  mul_sloppy_unnorm:   " << mean_sloppy/N << " +/- " << std_sloppy/N 
              << " (" << 100*std_sloppy/mean_sloppy << "%) ns/op\n";
    std::cout << "  speedup (norm/un):   " << mean_acc_norm/mean_acc_un << "x\n";
    std::cout << "  speedup (norm/sloppy): " << mean_acc_norm/mean_sloppy << "x\n";
    std::cout << "\n";
    
    save_results<N>(times_acc_norm, times_acc_un, times_sloppy_un);
}

int main() {
    std::cout << "Performance test\n";
    std::cout << "measurements=" << MEASUREMENTS << ", warmups=" << WARMUPS << ", real=" << REAL << "\n\n";
    
    std::ofstream out("results/perf_times.csv");
    out << "array_size,iteration,time_acc_norm,time_acc_un,time_sloppy_un\n";
    out.close();
    
    run_benchmark<100>();
    run_benchmark<1000>();
    run_benchmark<5000>();
    run_benchmark<10000>();
    run_benchmark<50000>();
    run_benchmark<100000>();
    run_benchmark<500000>();
    run_benchmark<1000000>();
    
    return 0;
}
