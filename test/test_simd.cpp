// DW<V> and XDW<V> with V a SIMD vector of float/double must give, in every lane, bit for bit what
// DW<lane> and XDW<lane> give on that lane's inputs. V is ext_vector_type on clang and vector_size on
// g++, as in MadGraph (clang mangles both kinds alike, so one translation unit can't mix them).

#include <cmath>
#include <cstdio>
#include <limits>
#include <random>

#include "../XDW.h"

using namespace XDW_ARTH;

#if defined( __clang__ )
template< typename S, int N >
using vec = S __attribute__( ( ext_vector_type( N ) ) );
#else
template< typename S, int N >
struct vec_helper
{
    typedef S type __attribute__( ( vector_size( N * sizeof( S ) ) ) );
};
template< typename S, int N >
using vec = typename vec_helper< S, N >::type;
#endif

constexpr int TRIALS = 20000;

static long failures = 0;

// Normalized DW with a random 2p-bit significand, sign and exponent in [-20, 20].
template< typename S >
static DW< S > random_dw( std::mt19937_64& rng )
{
    std::uniform_real_distribution< S > mantissa( S( 1 ), S( 2 ) ), unit( S( -1 ), S( 1 ) );
    std::uniform_int_distribution< int > exponent( -20, 20 );
    S h = std::ldexp( mantissa( rng ), exponent( rng ) );
    if( rng() & 1 )
        h = -h;
    const S l = std::ldexp( unit( rng ), std::ilogb( h ) - std::numeric_limits< S >::digits );
    const S hi = h + l;
    return DW< S >( hi, l - ( hi - h ) );
}

template< typename V, typename S = lane_t< V >, int N = lanes< V > >
static DW< V > pack( const DW< S > ( &x )[ N ] )
{
    V h, l;
    for( int i = 0; i < N; ++i ) {
        h[ i ] = x[ i ].hi();
        l[ i ] = x[ i ].lo();
    }
    return DW< V >( h, l );
}

template< typename V, AddMode A, NormMode Nm, typename S = lane_t< V >, int N = lanes< V > >
static long check_modes( const XDW< V >& z, const XDW< V >& w, const XDW< S > ( &zs )[ N ], const XDW< S > ( &ws )[ N ] )
{
    long bad = 0;
    const XDW< V > m = XDW< V >::template mul< A, Nm >( z, w );
    const XDW< V > d2 = XDW< V >::template div< DivMode::Div2, A, Nm >( z, w );
    const XDW< V > d3 = XDW< V >::template div< DivMode::Div3, A, Nm >( z, w );
    const DW< V > n = norm< V, A, Nm >( z );
    for( int i = 0; i < N; ++i ) {
        bad += !( m.lane( i ) == XDW< S >::template mul< A, Nm >( zs[ i ], ws[ i ] ) );
        bad += !( d2.lane( i ) == XDW< S >::template div< DivMode::Div2, A, Nm >( zs[ i ], ws[ i ] ) );
        bad += !( d3.lane( i ) == XDW< S >::template div< DivMode::Div3, A, Nm >( zs[ i ], ws[ i ] ) );
        bad += !( n.lane( i ) == norm< S, A, Nm >( zs[ i ] ) );
    }
    return bad;
}

template< typename V >
static void run( const char* name )
{
    using S = lane_t< V >;
    constexpr int N = lanes< V >;
    std::mt19937_64 rng( 7 );
    long bad = 0, total = 0;

    for( int t = 0; t < TRIALS; ++t ) {
        DW< S > xs[ N ], ys[ N ], us[ N ], vs[ N ];
        XDW< S > zs[ N ], ws[ N ];
        for( int i = 0; i < N; ++i ) {
            xs[ i ] = random_dw< S >( rng );
            ys[ i ] = random_dw< S >( rng );
            us[ i ] = random_dw< S >( rng );
            vs[ i ] = random_dw< S >( rng );
        }
        // Every fourth trial, lane 0 of y shares x's hi, so the comparisons are decided by lo.
        if( t % 4 == 0 )
            ys[ 0 ] = DW< S >( xs[ 0 ].hi(), xs[ 0 ].lo() * S( 0.5 ) );
        for( int i = 0; i < N; ++i ) {
            zs[ i ] = XDW< S >( xs[ i ], ys[ i ] );
            ws[ i ] = XDW< S >( us[ i ], vs[ i ] );
        }
        const DW< V > x = pack< V >( xs ), y = pack< V >( ys ), u = pack< V >( us ), v = pack< V >( vs );
        const XDW< V > z( x, y ), w( u, v );

        const DW< V > add = x + y, sub = x - y, mul = x * y, div = x / y, neg = -x;
        const DW< V > mixed = ( x + 0.5 ) * 2 - 1.0 / y;
        const auto lt = x < y, le = x <= y, gt = x > y, ge = x >= y, eq = x == y, ne = x != y;
        const XDW< V > ops = z * w / w + z - w;
        const XDW< V > zx = z * x, xz = x * z, zdx = z / x, zpx = z + x, xmz = x - z, zh = z * 0.5 + 1;
        const XDW< V > cz = conj( z );
        const auto zeq = z == w;
        for( int i = 0; i < N; ++i ) {
            const DW< S > a = xs[ i ], b = ys[ i ];
            bad += !( add.lane( i ) == a + b ) + !( sub.lane( i ) == a - b ) + !( mul.lane( i ) == a * b )
                 + !( div.lane( i ) == a / b ) + !( neg.lane( i ) == -a )
                 + !( mixed.lane( i ) == ( a + 0.5 ) * 2 - 1.0 / b );
            bad += ( ( lt[ i ] != 0 ) != ( a < b ) ) + ( ( le[ i ] != 0 ) != ( a <= b ) ) + ( ( gt[ i ] != 0 ) != ( a > b ) )
                 + ( ( ge[ i ] != 0 ) != ( a >= b ) ) + ( ( eq[ i ] != 0 ) != ( a == b ) ) + ( ( ne[ i ] != 0 ) != ( a != b ) );
            const XDW< S > c = zs[ i ], d = ws[ i ];
            bad += !( ops.lane( i ) == c * d / d + c - d );
            bad += !( zx.lane( i ) == c * a ) + !( xz.lane( i ) == a * c ) + !( zdx.lane( i ) == c / a )
                 + !( zpx.lane( i ) == c + a ) + !( xmz.lane( i ) == a - c ) + !( zh.lane( i ) == c * 0.5 + 1 );
            bad += !( cz.lane( i ) == conj( c ) ) + !( real( z ).lane( i ) == real( c ) ) + !( imag( z ).lane( i ) == imag( c ) );
            bad += ( ( zeq[ i ] != 0 ) != ( c == d ) );
        }
        bad += check_modes< V, AddMode::Madd, NormMode::Normalized >( z, w, zs, ws )
             + check_modes< V, AddMode::Madd, NormMode::Unnormalized >( z, w, zs, ws )
             + check_modes< V, AddMode::Accurate, NormMode::Normalized >( z, w, zs, ws )
             + check_modes< V, AddMode::Accurate, NormMode::Unnormalized >( z, w, zs, ws )
             + check_modes< V, AddMode::Sloppy, NormMode::Normalized >( z, w, zs, ws )
             + check_modes< V, AddMode::Sloppy, NormMode::Unnormalized >( z, w, zs, ws );
        total += N * ( 6 + 6 + 1 + 6 + 3 + 1 + 6 * 4 );

        // Broadcasts: a plain number, a scalar DW and a scalar XDW in every lane.
        const DW< V > bx( xs[ 0 ] ), b01( 0.1 );
        const XDW< V > bz( zs[ 0 ] ), b12( 0.1, 2 );
        for( int i = 0; i < N; ++i )
            bad += !( bx.lane( i ) == xs[ 0 ] ) + !( b01.lane( i ) == DW< S >( 0.1 ) ) + !( bz.lane( i ) == zs[ 0 ] )
                 + !( b12.lane( i ) == XDW< S >( 0.1, 2 ) );
        total += 4 * N;
    }
    std::printf( "%-24s %ld / %ld lane results differ from the scalar DW/XDW\n", name, bad, total );
    failures += bad;
}

int main()
{
    run< vec< double, 2 > >( "double x2" );
    run< vec< double, 4 > >( "double x4" );
    run< vec< double, 8 > >( "double x8" );
    run< vec< float, 4 > >( "float x4" );
    run< vec< float, 8 > >( "float x8" );
    run< vec< float, 16 > >( "float x16" );
    return failures == 0 ? 0 : 1;
}
