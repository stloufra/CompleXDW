#ifndef COMBO_MUL_H
#define COMBO_MUL_H

#include <array>

#include "ComplexDouble.h"

// All 6 (AddMode, NormMode) combinations ComplexDouble::mul<Add,Norm> can be instantiated with.
// Shared between test_complex_dw_conditioning.cpp and test_complex_dw_conditioning_binned.cpp
// so both stay in sync on combo order/naming.
constexpr int N_COMBOS = 6;
constexpr const char* COMBO_NAMES[N_COMBOS] = {
    "Madd/Normalized", "Madd/Unnormalized",
    "Accurate/Normalized", "Accurate/Unnormalized",
    "Sloppy/Normalized", "Sloppy/Unnormalized",
};

inline std::array<ComplexDouble<double>, N_COMBOS>
mul_all_combos(const ComplexDouble<double>& a, const ComplexDouble<double>& b)
{
    using namespace XDW_ARTH;
    return {
        ComplexDouble<double>::mul<AddMode::Madd, NormMode::Normalized>(a, b),
        ComplexDouble<double>::mul<AddMode::Madd, NormMode::Unnormalized>(a, b),
        ComplexDouble<double>::mul<AddMode::Accurate, NormMode::Normalized>(a, b),
        ComplexDouble<double>::mul<AddMode::Accurate, NormMode::Unnormalized>(a, b),
        ComplexDouble<double>::mul<AddMode::Sloppy, NormMode::Normalized>(a, b),
        ComplexDouble<double>::mul<AddMode::Sloppy, NormMode::Unnormalized>(a, b),
    };
}

#endif  //COMBO_MUL_H
