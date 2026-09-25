#ifndef RANDOM_DW_H
#define RANDOM_DW_H

#include <cmath>
#include <concepts>
#include <limits>
#include <random>

#include "ComplexDouble.h"

// Normalized double-word value with a random ~2p-bit significand, random sign and exponent in
// [-20, 20]: small enough that no operand or intermediate of a mul/div is subnormal, even for float.
template <std::floating_point T>
void random_dw(std::mt19937_64& rng, T* hi, T* lo)
{
    std::uniform_real_distribution<T> mantissa(T(1), T(2));
    std::uniform_real_distribution<T> unit(T(-1), T(1));
    std::uniform_int_distribution<int> exponent(-20, 20);

    T h = std::ldexp(mantissa(rng), exponent(rng));
    if (rng() & 1) h = -h;
    // |l| < ulp(h)/2, so (h, l) is already normalized; Fast2Sum makes it exact on ties too.
    T l = std::ldexp(unit(rng), std::ilogb(h) - std::numeric_limits<T>::digits);
    *hi = h + l;
    *lo = l - (*hi - h);
}

template <std::floating_point T>
ComplexDouble<T> random_dw_complex(std::mt19937_64& rng)
{
    T rh, rl, ih, il;
    random_dw(rng, &rh, &rl);
    random_dw(rng, &ih, &il);
    return ComplexDouble<T>(rh, rl, ih, il);
}

#endif  //RANDOM_DW_H
