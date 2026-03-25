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
    auto c_fast = mul_fast(a, b );
    std::cout << "Multiplication done" << std::endl;
    std::setprecision(16);
    std::cout << "c = (" << c.re_h() << ", " << c.re_l() << ") + i(" << c.im_h() << ", " << c.im_l() << ")" << std::endl;
    std::cout << "c_fast = (" << c_fast.re_h() << ", " << c_fast.re_l() << ") + i(" << c_fast.im_h() << ", " << c_fast.im_l() << ")" << std::endl;
    return 0;
}
