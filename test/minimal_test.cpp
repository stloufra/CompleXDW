#include <iomanip>
#include <iostream>
#include "../ComplexDouble.h"

using namespace XDW_ARTH;

int main() {
    std::cout << "Creating ComplexDouble<double>..." << std::endl;
    ComplexDouble<double> a(1.0, 0.0, 1.0, 0.0);
    std::cout << "Created a" << std::endl;
    ComplexDouble<double> b(2.0, 0.0, 3.0, 0.0);
    std::cout << "Created b" << std::endl;
    auto c = a * b;
    auto c_sloppy = mul_sloppy_unnnorm(a, b );
    auto c_accurate = mul_accurate_unnnorm(a, b );
    std::cout << "Multiplication done" << std::endl;
    std::setprecision(16);
    std::cout << "c = (" << c.re_h() << ", " << c.re_l() << ") + i(" << c.im_h() << ", " << c.im_l() << ")" << std::endl;
    std::cout << "c_sloppy_unnorm = (" << c_sloppy.re_h() << ", " << c_sloppy.re_l() << ") + i(" << c_sloppy.im_h() << ", " << c_sloppy.im_l() << ")" << std::endl;
    std::cout << "c_accurate_unnorm = (" << c_accurate.re_h() << ", " << c_accurate.re_l() << ") + i(" << c_accurate.im_h() << ", " << c_accurate.im_l() << ")" << std::endl;
    return 0;
}
