#ifndef COMBO_DIV_H
#define COMBO_DIV_H

#include <array>

#include "ComplexDouble.h"

// All 12 (DivMode, AddMode, NormMode) combinations ComplexDouble::div<Div,Add,Norm> can be instantiated with.
constexpr int N_DIV_COMBOS = 12;
constexpr const char* DIV_COMBO_NAMES[N_DIV_COMBOS] = {
    "Div2/Madd/Normalized", "Div2/Madd/Unnormalized",
    "Div2/Accurate/Normalized", "Div2/Accurate/Unnormalized",
    "Div2/Sloppy/Normalized", "Div2/Sloppy/Unnormalized",
    "Div3/Madd/Normalized", "Div3/Madd/Unnormalized",
    "Div3/Accurate/Normalized", "Div3/Accurate/Unnormalized",
    "Div3/Sloppy/Normalized", "Div3/Sloppy/Unnormalized",
};

inline std::array<ComplexDouble<double>, N_DIV_COMBOS>
div_all_combos(const ComplexDouble<double>& a, const ComplexDouble<double>& b)
{
    using namespace XDW_ARTH;
    using C = ComplexDouble<double>;
    return {
        C::div<DivMode::Div2, AddMode::Madd, NormMode::Normalized>(a, b),
        C::div<DivMode::Div2, AddMode::Madd, NormMode::Unnormalized>(a, b),
        C::div<DivMode::Div2, AddMode::Accurate, NormMode::Normalized>(a, b),
        C::div<DivMode::Div2, AddMode::Accurate, NormMode::Unnormalized>(a, b),
        C::div<DivMode::Div2, AddMode::Sloppy, NormMode::Normalized>(a, b),
        C::div<DivMode::Div2, AddMode::Sloppy, NormMode::Unnormalized>(a, b),
        C::div<DivMode::Div3, AddMode::Madd, NormMode::Normalized>(a, b),
        C::div<DivMode::Div3, AddMode::Madd, NormMode::Unnormalized>(a, b),
        C::div<DivMode::Div3, AddMode::Accurate, NormMode::Normalized>(a, b),
        C::div<DivMode::Div3, AddMode::Accurate, NormMode::Unnormalized>(a, b),
        C::div<DivMode::Div3, AddMode::Sloppy, NormMode::Normalized>(a, b),
        C::div<DivMode::Div3, AddMode::Sloppy, NormMode::Unnormalized>(a, b),
    };
}

#endif  //COMBO_DIV_H
