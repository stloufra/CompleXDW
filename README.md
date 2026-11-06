# CompleXDW

> *It's the little differences. In Europe they call it double, here it's `DW<float>`.*\
> *And complex quad? They call it Le `XDW<double>`.*

CompleXDW is a header-only C++20 library for complex arithmetic in double-word precision: each real
and imaginary part is an unevaluated non-overlapping sum `hi + lo` of two floating-point numbers.

| Base type | Precision of each part | `u` | `u²` |
|---|---|---|---|
| `double` (double-double) | ~106 bits, emulated quad | 2⁻⁵³ | 2⁻¹⁰⁶ ≈ 1.2*10⁻³² |
| `float` (float-float) | ~48 bits, emulated double | 2⁻²⁴ | 2⁻⁴⁸ ≈ 3.6*10⁻¹⁵ |

- `DW<T>` is a real double-word number
- `XDW<T>` is a complex double-word number
- `T` can be `float`, `double`, or a compiler SIMD vector of either
- Everything is `constexpr` runs on CUDA and HIP devices
- The accuracy/speed trade-off of complex `*`, `/` and `norm` is chosen at compile time, either globally or per call

## Quick start

```cpp
#include "XDW.h"

using namespace XDW_ARTH;

using Complex = XDW< float >;               // float-float, ~48 bits per part
using Real = DW< float >;

Complex a( 1.0, 2.0 ), b( 3.0, -1.0 );

Complex p = a * b;                          // default modes
Complex q = a / b;
Real n = norm( a );                         // |a|^2, no sqrt
Real r = sqrt( n );

Complex fast = Complex::mul< AddMode::Sloppy, NormMode::Unnormalized >( a, b );
Complex div2 = Complex::div< DivMode::Div2 >( a, b );

Complex z = a * 0.5f + 2.f;                 // a float is one FP operand: DWTimesFP3, DWPlusFP
Complex y = a * 0.1;                        // a double is split into a DW<float> first
std::complex< double > c( p );              // explicit, exact into the wider type
Complex from_std = std::complex< double >( 1, 2 );  // any type with real()/imag()
```

Add the repository root to the include path. There is nothing to build.

## Requirements

> *Say `fast math` again. I `double` dare you.*

- C++20
- Round-to-nearest IEEE arithmetic without excess precision. The library refuses to compile under `-ffast-math`/`-Ofast` or with `FLT_EVAL_METHOD == 2` (x87)
- `-ffp-contract=off`. Fusing into FMA changes the error-free transforms
- Hardware FMA (where needed), e.g. `-march=native`, or `-mfma` on x86-64. Without it every `fma` is a libm call, and the library warns (slow). Define `XDW_ALLOW_SOFTWARE_FMA` to silence the warning.

## Types

### `DW<T>`: real double-word

| Operation | Algorithm | Flops | Relative error |
|---|---|---|---|
| `x + y`, `x - y` | `MaddDWPlusDW` | 20 | 2 u² |
| `x * y` | `DWTimesDW2` | 8 | 5 u² |
| `x / y` | `DWDivDW2` | 18 | 15 u² + 56 u³ |
| `x + y`, `x - y` (`y` plain, exact in `T`) | `DWPlusFP` | 10 | 2 u² |
| `x * y` (`y` plain, exact in `T`) | `DWTimesFP3` | 6 | 2 u² |
| `x / y` (`y` plain, exact in `T`) | `DWDivFP3` | 10 | 3 u² |
| `sqrt(x)` | `DWSqrt` | 1 sqrt + 7 | 25/8 u² |
| `==` `!=` `<` `<=` `>` `>=`, `min`, `max`, `abs`, `signbit`, `select` | exact | – | – |

- **Construction:** a wider value is split into `hi` and `lo`, so `DW<float>(double)` keeps 48 of the 53 bits. Conversion back is explicit and exact into a wider type: `DW<float>` > `double` gives `hi + lo`. Otherwise it gives `hi`.
- **Precision casts:** `DW<double>` <-> `DW<float>` conversions are explicit. Going up is exact. Going down rounds to 48 bits.

### `XDW<T>`: complex double-word

| Operation | Flops | Error per component |
|---|---|---|
| `z + w`, `z - w` | 40 | 2 u² (Madd, independent of `AddMode`) |
| `z * w` | 42–72 | see [complex multiplication](#complex-multiplication) |
| `z / w` | 97–135 | see [complex division](#complex-division) |
| `norm(z)` = \|z\|² | 19–34 | see [norm](#norm) |
| `z + x`, `z - x` (`x` real DW) | 20 | 2 u² |
| `z * x` | 16 | 5 u² |
| `z / x` | 36 | 15 u² + 56 u³ |
| `x / z` | full complex division | – |
| `z + y`, `z - y` (`y` plain, exact in `T`) | 10 | 2 u² |
| `z * y` | 12 | 2 u² |
| `z / y` | 20 | 3 u² |
| `y / z` | full complex division | – |
| `conj`, `real`, `imag`, `-z`, `==`, `select` | exact | – |

`XDW<T>` can be constructed from:

- plain numbers: `XDW<T>(re)` and `XDW<T>(re, im)`;
- two `DW<T>` parts;
- the four words: `XDW<T>(re_h, re_l, im_h, im_l)`;
- any complex type exposing `real()` and `imag()`, such as `std::complex` or `thrust::complex`.

### Mixed operands

> *Where is my double?*\
> *I had to split the double.*\
> *Whose 53 bits are these?*\
> *It's not 53 bits, it's 42.*\
> *Whose 42 is it?*\
> *`DW<float>`'s.*\
> *Who is `DW<float>`?*\
> *`DW<float>` is dead, baby. `DW<float>` is dead.*\
> — the dialog of lossy conversion

- **Plain number exact in `T`:** a number that converts to the lane type without rounding is used as one floating-point operand: `DWPlusFP`, `DWTimesFP3` and `DWDivFP3`, on each part of an `XDW`
- **Any other plain number** is split into a `DW` first, without rounding, and the DW–DW operation runs
- **Scalar next to vector:** a scalar `DW<S>`/`XDW<S>` next to a vector `DW<V>`/`XDW<V>` with lane type `S` is broadcast to every lane

### SIMD vectors

```cpp
using float8 = float __attribute__( ( ext_vector_type( 8 ) ) );   // clang
// typedef float float8 __attribute__( ( vector_size( 32 ) ) );   // g++

using ComplexV = XDW< float8 >;          // 8 independent complex numbers
using RealV = DW< float8 >;

float8 re = ..., im = ...;               // e.g. loaded from arrays
ComplexV z( RealV( re, float8{} ), RealV( im, float8{} ) );

ComplexV w = z * Complex( 0.5f, 1.f ) + 1.f;   // scalar Complex and float go to every lane
Complex lane2 = w.lane( 2 );
auto mask = real( w ) < real( z );       // lane mask
ComplexV smaller = select( mask, w, z );
```

Each lane gives bit-for-bit the same result as the scalar type on that lane's inputs. Vector types are host only.

## Modes

Complex `*`, `/` and `norm` are built from real mul-adds `a*b +(-) c*d`. Three independent choices control their cost and accuracy:

| Mode | Values | Meaning |
|---|---|---|
| `AddMode` | `Accurate`, `Madd`, `Sloppy` | DW addition used to combine the two products: `AccurateDWPlusDW` (20 flops, 3 u² + 13 u³), `MaddDWPlusDW` (20 flops, 2 u²), or `SloppyDWPlusDW` (11 flops, accurate only without cancellation) |
| `NormMode` | `Normalized`, `Unnormalized` | Whether each product is renormalized (`DWTimesDW2`, 8 flops) or passed on unnormalized (`DWTimesDW2Unnorm`, 5 flops, skips the final Fast2Sum, the Winston Wolf of double-word arithmetic) |
| `DivMode` | `Div2`, `Div3` | Complex division: two `DWDivDW2` divisions, or one shared Newton-refined reciprocal (`DWRecip`) times two `DWTimesDW2`|

See our [paper](https://arxiv.org/abs/2611.08371) to be convinced why skipping normalization in complex operations is not the worst idea.

**Defaults:** `Madd`, `Normalized`, `Div3`.

To set the defaults for a translation unit:

```
-DXDW_ADD_MODE=A|M|S   -DXDW_NORM_MODE=N|U   -DXDW_DIV_MODE=2|3
```

To set the modes for one call, pass template arguments: `XDW<T>::mul<Add, Norm>(a, b)`, `XDW<T>::div<Div, Add, Norm>(a, b)` and `norm<T, Add, Norm>(z)`. Invalid macro values are rejected at compile time.

### Conditioning

A mul-add `p + q`, with `p = a*b` and `q = c*d`, cannot be more accurate than its conditioning allows:

```
K = (|p| + |q|) / |p + q|      (K = 1 without cancellation)
```

The complex bounds below are per component, relative error ≤ `bound * K * u²`. Each component uses its own `K`: `ac − bd` for the real part of a product, and `ad + bc` for the imaginary part.

### Complex multiplication

`z * w`: two mul-adds.

| AddMode | NormMode | Flops | Bound (K*u²) |
|---|---|---|---|
| Madd | Normalized | 72 | 7 |
| Accurate | Normalized | 72 | 8 |
| Sloppy | Normalized | 54 | 8  |
| Madd | Unnormalized | 60 | 8 |
| Accurate | Unnormalized | 60 | 10 |
| Sloppy | Unnormalized | 42 | 12  |

When the conditioning is bad, none of the digits may be correct. In that case the answer is always 42.

### Complex division

`z / w = (ac + bd)/(c² + d²) + i (bc − ad)/(c² + d²)`

The work is split into three parts:

- **Denominator:** `c² + d²` cannot cancel, so it always uses `Sloppy` addition, with the selected `NormMode`. This costs 25 flops and 6 u² when normalized, or 19 flops and 10 u² unnormalized.
- **Numerators:** two mul-adds with the selected modes.
- **Division:** `Div2` costs 2 × 18 flops (15 u² + 56 u³). `Div3` costs 22 + 2 × 8 flops (9.8 u²).

The bound per component is the sum of the numerator, denominator and division bounds, scaled by the numerator's `K`.

| AddMode | NormMode | Flops Div2 | Flops Div3 | Bound Div2 (K*u²) | Bound Div3 (K*u²) |
|---|---|---|---|---|---|
| Madd | Normalized | 133 | 135 | 28 | 22.8 |
| Accurate | Normalized | 133 | 135 | 29 | 23.8 |
| Sloppy | Normalized | 115 | 117 | 29  | 23.8  |
| Madd | Unnormalized | 115 | 117 | 33 | 27.8 |
| Accurate | Unnormalized | 115 | 117 | 35 | 29.8 |
| Sloppy | Unnormalized | 97 | 99 | 37  | 31.8  |

The Div2 bounds also carry the + 56 u³ term from `DWDivDW2`.

### Norm

`norm(z) = re² + im²` (`DWPowAdd`). It cannot cancel, so `K = 1`.

| AddMode | NormMode | Flops | Bound (u²) |
|---|---|---|---|
| Madd | Normalized | 34 | 7 |
| Accurate | Normalized | 34 | 8 |
| Sloppy | Normalized | 25 | 6 |
| Madd | Unnormalized | 28 | 8 |
| Accurate | Unnormalized | 28 | 10 |
| Sloppy | Unnormalized | 19 | 10 |

## Building blocks

The double-word algorithms live in `XDW_ARTH` (`src/XDWarith.h`), and the error-free transforms in `src/XDWerrorFree.h`. They are usable directly on raw `(hi, lo)` words.

| Algorithm | Computes | Flops | Relative error (u²) |
|---|---|---|---|
| `SloppyDWPlusDW` | x + y | 11 | none under cancellation |
| `AccurateDWPlusDW` | x + y | 20 | 3 + 13 u |
| `MaddDWPlusDW` | x + y | 20 | 2 |
| `DWPlusFP` | x + y, y one FP number | 10 | 2 |
| `DWTimesDW2` | x * y | 8 | 5 |
| `DWTimesDW2Unnorm` | x * y, unnormalized | 5 | 5 |
| `DWTimesDW3` | x * y | 9 | 4 |
| `DWTimesFP1` | x * y, y one FP number | 10 | 3/2 + 4 u |
| `DWTimesFP3` | x * y, y one FP number | 6 | 2 |
| `DWDivFP3` | x / y, y one FP number | 10 | 3 |
| `DWPow2` | x² | 7 | 5 |
| `DWPow2Unnorm` | x², unnormalized | 4 | 5 |
| `DWDivDW2` | x / y | 18 | 15 + 56 u |
| `DWRecip` | 1 / y | 22 | – |
| `DWDivDW3` | x / y | 31 | 9.8 |
| `DWSqrt` | √x | 1 sqrt + 7 | 25/8 |
| `DWMulAdd_<Add>_<Norm>` | a*b + c*d | 21–36 | 7–12 * K |

**How flops are counted:** each `+ − × ÷` and each FMA counts as 1, and negation is free. That makes TwoSum 6, Fast2Sum 3 and TwoProd 2 (with FMA).

> *Gets medieval on TwoProd: Dekker's split, straight out of 1971*
> 
**Dekker splitting:** defining `XDW_DEKKER_TWO_PROD` replaces the FMA in TwoProd with Dekker's split. This is host only, and it raises every multiplication's cost.

## Source

The error analysis and proofs of the bounds are in:

F. Stloukal and F. Jézéquel, *Lord of the Precision, the Two Floats: Double-Word Arithmetic for Complex Numbers*, [arXiv:2611.08371](https://arxiv.org/abs/2611.08371), 2026.

If you use CompleXDW, please cite:

```bibtex
@misc{stloukal2026lord,
  title         = {Lord of the Precision, the Two Floats: Double-Word Arithmetic for Complex Numbers},
  author        = {Stloukal, Franti{\v{s}}ek and J{\'e}z{\'e}quel, Fabienne},
  year          = {2026},
  eprint        = {2611.08371},
  archivePrefix = {arXiv},
  primaryClass  = {cs.MS},
  url           = {https://arxiv.org/abs/2611.08371}
}
```

## GPU

All scalar operations are `__host__ __device__` under `nvcc` and `hipcc`. On the device they use the correctly rounded `__dadd_rn`/`__fmul_rn`/`__fma_rn`/… intrinsics, so the compiler cannot contract or reorder them. With `NDEBUG`, everything is force-inlined.

## Tests

The unit tests compare every algorithm and mode against MPFR, and check that SIMD lanes match the scalar results bit for bit.

```bash
make -C test/unit test
```

| Option | Default |
|---|---|
| `CXX=g++-16` | system `c++` |
| `OPT=-O3` | `-O2` |
| `ARCH=` | `-march=native` |
| `CXXFLAGS="..."` | appended last |
| `MPFR_PREFIX=/path` | Homebrew / system paths |

The tests need MPFR and GMP. CI runs them with g++ and clang++ on Ubuntu, and with clang++ on macOS.


## License

MIT, see [LICENSE](LICENSE).
