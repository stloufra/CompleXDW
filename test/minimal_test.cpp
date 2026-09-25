#include <iomanip>
#include <iostream>
#include "../XDW.h"

using namespace XDW_ARTH;

int main() {
    std::cout << "Creating XDW<double>..." << std::endl;
    XDW<double> a(1.0, 0.0, 1.0, 0.0);
    std::cout << "Created a" << std::endl;
    XDW<double> b(2.0, 0.0, 3.0, 0.0);
    std::cout << "Created b" << std::endl;
    auto c = a * b;
    auto c_sloppy = XDW<double>::mul<AddMode::Sloppy, NormMode::Unnormalized>(a, b);
    auto c_accurate = XDW<double>::mul<AddMode::Madd, NormMode::Unnormalized>(a, b);
    std::cout << "Multiplication done" << std::endl;
    std::setprecision(16);
    std::cout << "c = (" << c.re_h() << ", " << c.re_l() << ") + i(" << c.im_h() << ", " << c.im_l() << ")" << std::endl;
    std::cout << "c_sloppy_unnorm = (" << c_sloppy.re_h() << ", " << c_sloppy.re_l() << ") + i(" << c_sloppy.im_h() << ", " << c_sloppy.im_l() << ")" << std::endl;
    std::cout << "c_accurate_unnorm = (" << c_accurate.re_h() << ", " << c_accurate.re_l() << ") + i(" << c_accurate.im_h() << ", " << c_accurate.im_l() << ")" << std::endl;
    return 0;
}
