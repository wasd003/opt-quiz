#include <cstdio>
#include <cstdint>
#include <type_traits>
#include <immintrin.h>

#ifndef KEWB_SIMD_HPP_DEFINED
#define KEWB_SIMD_HPP_DEFINED

#include <cstdio>
#include <cstdint>

#include <complex>
#include <limits>
#include <type_traits>

#ifdef __OPTIMIZE__
    #include <immintrin.h>
    #define KEWB_FORCE_INLINE   __attribute__((__always_inline__)) inline
#else
    #define __OPTIMIZE__
    #include <immintrin.h>
    #undef __OPTIMIZE__
    #define KEWB_FORCE_INLINE   inline
#endif

namespace simd {

using rd128 = __m128d;
using rf128 = __m128;
using ri128 = __m128i;

using rd256 = __m256d;
using rf256 = __m256;
using ri256 = __m256i;

using double_512 = __m512d;
using float_512 = __m512;
using integer_512 = __m512i;

using r512d = __m512d;
using r512f = __m512;
using r512i = __m512i;

using m256 = __m256i;
using m512 = uint32_t;

void
print_vals(char const* pname, double const* pvals, int n)
{
    if (pname != nullptr)
    {
        printf("reg %s:\n", pname);
    }

    for (int i = 0;  i < n;  ++i)
    {
        printf("%6.1f", pvals[i]);
    }
    printf("\n");
    fflush(stdout);
}

void
print_vals(char const* pname, float const* pvals, int n)
{
    if (pname != nullptr)
    {
        printf("reg %s:\n", pname);
    }

    for (int i = 0;  i < n;  ++i)
    {
        printf("%6.1f", pvals[i]);
    }
    printf("\n");
    fflush(stdout);
}

void
print_vals(char const* pname, int32_t const* pvals, int n)
{
    if (pname != nullptr)
    {
        printf("reg %s:\n", pname);
    }

    for (int i = 0;  i < n;  ++i)
    {
        printf("%6d", pvals[i]);
    }
    printf("\n");
    fflush(stdout);
}


void
print_reg(char const* pname, uint32_t i);

//------
//
void
print_reg(char const* pname, rd128 r)
{
    double  vals[2];

    _mm_storeu_pd(&vals[0], r);
    print_vals(pname, vals, 2);
}

void
print_reg(char const* pname, rf128 r)
{
    float   vals[4];

    _mm_storeu_ps(&vals[0], r);
    print_vals(pname, vals, 4);
}

void
print_reg(char const* pname, ri128 r)
{
    int32_t vals[4];

    _mm_storeu_epi32(&vals[0], r);
    print_vals(pname, vals, 4);
}

//------
//
void
print_reg(char const* pname, rd256 r)
{
    double  vals[4];

    _mm256_storeu_pd(&vals[0], r);
    print_vals(pname, vals, 4);
}

void
print_reg(char const* pname, rf256 r)
{
    float   vals[8];

    _mm256_storeu_ps(&vals[0], r);
    print_vals(pname, vals, 8);
}

void
print_reg(char const* pname, ri256 r)
{
    int32_t vals[8];

    _mm256_storeu_si256((ri256*) &vals[0], r);
    print_vals(pname, vals, 8);
}

//------
//
void
print_reg(char const* pname, double_512 r)
{
    double  vals[8];

    _mm512_storeu_pd(&vals[0], r);
    print_vals(pname, vals, 8);
}

void
print_reg(char const* pname, float_512 r)
{
    float   vals[16];

    _mm512_storeu_ps(&vals[0], r);
    print_vals(pname, vals, 16);
}

void
print_reg(char const* pname, integer_512 r)
{
    int32_t vals[16];

    _mm512_storeu_epi32(&vals[0], r);
    print_vals(pname, vals, 16);
}

void
print_mask(char const* pname, uint32_t mask, int bits)
{
    if (pname != nullptr)
    {
        printf("mask %s:\n", pname);
    }

    uint32_t    probe = 1;

    for (int i = 0;  i < bits;  ++i, probe <<= 1)
    {
        printf("%6d", (mask & probe) ? 1 : 0);
    }
    printf("\n");
    fflush(stdout);
}

void
print_mask(char const* pname, __m256i mask, int)
{
    if (pname != nullptr)
    {
        printf("mask %s:\n", pname);
    }
    print_reg(nullptr, mask);
}

#define PRINT_REG(R)        simd::print_reg(#R, R)
#define PRINT_MASK(M)       print_mask(#M, M, 16)
#define PRINT_MASK8(M)      print_mask(#M, M, 8)
#define PRINT_MASK16(M)     print_mask(#M, M, 16)
#define PRINT_LINE()        printf("\n");


/**
 * SEQ: 0
 * @load_value
 * load a immediate value into SIMD register
 * @load_values
 * load a set of immediate values into SIMD register
 */
KEWB_FORCE_INLINE float_512 load_value(float immediate_val) {
    return _mm512_set1_ps(immediate_val);
}

KEWB_FORCE_INLINE integer_512 load_value(int32_t immediate_val) {
    return _mm512_set1_epi32(immediate_val);
}

template<int A, int B, int C, int D, int E, int F, int G, int H,
         int I, int J, int K, int L, int M, int N, int O, int P>
KEWB_FORCE_INLINE integer_512
load_values() {
    return _mm512_setr_epi32(A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P);
}

KEWB_FORCE_INLINE integer_512
load_values(int a, int b, int c, int d, int e, int f, int g, int h,
            int i, int j, int k, int l, int m, int n, int o, int p) {
    return _mm512_setr_epi32(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p);
}

KEWB_FORCE_INLINE float_512
load_values(float a, float b, float c, float d, float e, float f, float g, float h,
            float i, float j, float k, float l, float m, float n, float o, float p)
{
    return _mm512_setr_ps(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p);
}


/**
 * SEQ: 1
 * @load_from
 * load data from memory into SIMD register
 */
KEWB_FORCE_INLINE double_512 load_from(const double *psrc) {
    return _mm512_loadu_pd(psrc);
}

KEWB_FORCE_INLINE float_512 load_from(const float* psrc) {
    return _mm512_loadu_ps(psrc);
}

KEWB_FORCE_INLINE integer_512 load_from(const int32_t* psrc) {
    return _mm512_loadu_epi32(psrc);
}

/**
 * SEQ: 2
 * @masked_load_from
 * load data from [memory] or [SIMD register/immediate value]
 *      - if mask bit is 1, load from memory
 *      - if mask bit is 0, load from SIMD register/immediate value
 */
KEWB_FORCE_INLINE float_512 masked_load_from(const float* psrc, float_512 simd_reg, uint32_t mask) {
    return _mm512_mask_loadu_ps(simd_reg, (__mmask16) mask, psrc);
}

KEWB_FORCE_INLINE float_512 masked_load_from(const float* psrc, float immediate_val, uint32_t mask) {
    return _mm512_mask_loadu_ps(_mm512_set1_ps(immediate_val), (__mmask16) mask, psrc);
}


/**
 * SEQ: 3
 * @store_to
 * store data from SIMD register to memory
 */
KEWB_FORCE_INLINE void store_to(void* pdst, integer_512 r) {
    _mm512_mask_storeu_epi32(pdst, (__mmask16) 0xFFFFu, r);
}

KEWB_FORCE_INLINE void store_to(void* pdst, float_512 r) {
    _mm512_mask_storeu_ps(pdst, (__mmask16) 0xFFFFu, r);
}

/**
 * SEQ: 4
 * @masked_store_to
 * store data to memory based on mask
 *    - if mask bit is 1, store from SIMD register
 *    - if mask bit is 0, leave memory unchanged
 */
KEWB_FORCE_INLINE void masked_store_to(void* pdst, float_512 r, uint32_t mask)
{
    _mm512_mask_storeu_ps(pdst, (__mmask16) mask, r);
}

/**
 * SEQ: 5
 * @blend
 * blend data from two SIMD registers based on mask
 *    - if mask bit is 1, use data from r1
 *    - if mask bit is 0, use data from r0
 */
KEWB_FORCE_INLINE float_512 blend(float_512 r0, float_512 r1, uint32_t mask) {
    return _mm512_mask_blend_ps((__mmask16) mask, r0, r1);
}


/**
 * SEQ: 6
 * @permute
 * export answer register based on r and perm
 *      answer[i] = r[perm[i]]
 */
KEWB_FORCE_INLINE float_512 permute(float_512 r, integer_512 perm) {
    return _mm512_permutexvar_ps(perm, r);
}

KEWB_FORCE_INLINE integer_512 permute(integer_512 r, integer_512 perm) {
    return _mm512_permutexvar_epi32(perm, r);
}

template<int A, int B, int C, int D, int E, int F, int G, int H,
         int I, int J, int K, int L, int M, int N, int O, int P>
KEWB_FORCE_INLINE integer_512 permute(integer_512 r0) {
    return _mm512_permutexvar_epi32(load_values<A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P>(), r0);
}

/**
 * SEQ: 7
 * @masked_permute
 * export answer register based on r and perm and mask
 *     if mask[i] == 1, then answer[i] = r[perm[i]]
 *     if mask[i] == 0, then answer[i] = other_reg[i]
 */
KEWB_FORCE_INLINE float_512 masked_permute(float_512 r, float_512 other_reg, integer_512 perm, uint32_t mask) {
    return _mm512_mask_permutexvar_ps(other_reg, (__mmask16) mask, perm, r);
}

///    if mask[i] == 1, then answer[i] = r[perm[i]]
///    if mask[i] == 0, then answer[i] = r[i]
KEWB_FORCE_INLINE float_512 masked_permute(float_512 r, integer_512 perm, uint32_t mask) {
    return _mm512_mask_permutexvar_ps(r, (__mmask16) mask, perm, r);
}

/**
 * SEQ: 8
 * @make_perm
 * create a permutation register based on immediate values
 */
template<unsigned... IDXS>
KEWB_FORCE_INLINE auto
make_perm()
{
    static_assert(sizeof...(IDXS) == 8  ||  sizeof...(IDXS) == 16);

    if constexpr (sizeof...(IDXS) == 8) {
        return _mm256_setr_epi32(IDXS...);
    } else {
        return load_values<IDXS...>();
    }
}




template<int BIAS, uint32_t MASK>
KEWB_FORCE_INLINE __m512i
make_shift_permutation()
{
    constexpr int32_t   a = ((BIAS + 0)  % 16) | ((MASK & 1u)        ? 0x10 : 0);
    constexpr int32_t   b = ((BIAS + 1)  % 16) | ((MASK & 1u << 1u)  ? 0x10 : 0);
    constexpr int32_t   c = ((BIAS + 2)  % 16) | ((MASK & 1u << 2u)  ? 0x10 : 0);
    constexpr int32_t   d = ((BIAS + 3)  % 16) | ((MASK & 1u << 3u)  ? 0x10 : 0);
    constexpr int32_t   e = ((BIAS + 4)  % 16) | ((MASK & 1u << 4u)  ? 0x10 : 0);
    constexpr int32_t   f = ((BIAS + 5)  % 16) | ((MASK & 1u << 5u)  ? 0x10 : 0);
    constexpr int32_t   g = ((BIAS + 6)  % 16) | ((MASK & 1u << 6u)  ? 0x10 : 0);
    constexpr int32_t   h = ((BIAS + 7)  % 16) | ((MASK & 1u << 7u)  ? 0x10 : 0);
    constexpr int32_t   i = ((BIAS + 8)  % 16) | ((MASK & 1u << 8u)  ? 0x10 : 0);
    constexpr int32_t   j = ((BIAS + 9)  % 16) | ((MASK & 1u << 9u)  ? 0x10 : 0);
    constexpr int32_t   k = ((BIAS + 10) % 16) | ((MASK & 1u << 10u) ? 0x10 : 0);
    constexpr int32_t   l = ((BIAS + 11) % 16) | ((MASK & 1u << 11u) ? 0x10 : 0);
    constexpr int32_t   m = ((BIAS + 12) % 16) | ((MASK & 1u << 12u) ? 0x10 : 0);
    constexpr int32_t   n = ((BIAS + 13) % 16) | ((MASK & 1u << 13u) ? 0x10 : 0);
    constexpr int32_t   o = ((BIAS + 14) % 16) | ((MASK & 1u << 14u) ? 0x10 : 0);
    constexpr int32_t   p = ((BIAS + 15) % 16) | ((MASK & 1u << 15u) ? 0x10 : 0);

    return _mm512_setr_epi32(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p);
}

template<int R>
KEWB_FORCE_INLINE __m512
rotate(__m512 r0)
{
    if constexpr ((R % 16) == 0)
    {
        return r0;
    }
    else
    {
        constexpr int    S = (R > 0) ? (16 - (R % 16)) : -R;
        constexpr int    A = (S + 0) % 16;
        constexpr int    B = (S + 1) % 16;
        constexpr int    C = (S + 2) % 16;
        constexpr int    D = (S + 3) % 16;
        constexpr int    E = (S + 4) % 16;
        constexpr int    F = (S + 5) % 16;
        constexpr int    G = (S + 6) % 16;
        constexpr int    H = (S + 7) % 16;
        constexpr int    I = (S + 8) % 16;
        constexpr int    J = (S + 9) % 16;
        constexpr int    K = (S + 10) % 16;
        constexpr int    L = (S + 11) % 16;
        constexpr int    M = (S + 12) % 16;
        constexpr int    N = (S + 13) % 16;
        constexpr int    O = (S + 14) % 16;
        constexpr int    P = (S + 15) % 16;

        return _mm512_permutexvar_ps(_mm512_setr_epi32(A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P), r0);
    }
}

template<int R>
KEWB_FORCE_INLINE __m512i
rotate(__m512i r0)
{
    if constexpr ((R % 16) == 0)
    {
        return r0;
    }
    else
    {
        constexpr int    S = (R > 0) ? (16 - (R % 16)) : -R;
        constexpr int    A = (S + 0) % 16;
        constexpr int    B = (S + 1) % 16;
        constexpr int    C = (S + 2) % 16;
        constexpr int    D = (S + 3) % 16;
        constexpr int    E = (S + 4) % 16;
        constexpr int    F = (S + 5) % 16;
        constexpr int    G = (S + 6) % 16;
        constexpr int    H = (S + 7) % 16;
        constexpr int    I = (S + 8) % 16;
        constexpr int    J = (S + 9) % 16;
        constexpr int    K = (S + 10) % 16;
        constexpr int    L = (S + 11) % 16;
        constexpr int    M = (S + 12) % 16;
        constexpr int    N = (S + 13) % 16;
        constexpr int    O = (S + 14) % 16;
        constexpr int    P = (S + 15) % 16;

        return _mm512_permutexvar_epi32(_mm512_setr_epi32(A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P), r0);
    }
}

template<int R>
KEWB_FORCE_INLINE __m512
rotate_down(__m512 r0)
{
    static_assert(R >= 0);
    return rotate<-R>(r0);
}

template<int R>
KEWB_FORCE_INLINE __m512
rotate_up(__m512 r0)
{
    static_assert(R >= 0);
    return rotate<R>(r0);
}

template<int S>
KEWB_FORCE_INLINE constexpr uint32_t
shift_down_blend_mask()
{
    static_assert(S >= 0  &&  S <= 16);
    return (0xFFFFu << (unsigned)(16 - S)) & 0xFFFFu;
}

template<int S>
KEWB_FORCE_INLINE void
in_place_shift_down_with_carry(__m512& lo, __m512& hi)
{
    static_assert(S >= 0  &&  S <= 16);

    constexpr uint32_t  zmask = (0xFFFFu >> (unsigned) S);
    constexpr uint32_t  bmask = ~zmask & 0xFFFFu;
    __m512i             perm  = make_shift_permutation<S, bmask>();

    lo = _mm512_permutex2var_ps(lo, perm, hi);
    hi = _mm512_maskz_permutex2var_ps((__mmask16) zmask, hi, perm, hi);
}


template<int S>
KEWB_FORCE_INLINE constexpr uint32_t
shift_up_blend_mask()
{
    static_assert(S >= 0  &&  S <= 16);
    return (0xFFFFu << (unsigned) S) & 0xFFFFu;
}

template<int S>
KEWB_FORCE_INLINE __m512
shift_up(__m512 r0)
{
    return blend(rotate_up<S>(r0), load_value(0), shift_up_blend_mask<S>());
}

template<int S>
KEWB_FORCE_INLINE __m512
shift_down(__m512 r0)
{
    return blend(rotate_down<S>(r0), load_value(0), shift_down_blend_mask<S>());
}

template<int S>
KEWB_FORCE_INLINE __m512
shift_up_with_carry(__m512 lo, __m512 hi)
{
    return blend(rotate_up<S>(lo), rotate_up<S>(hi), shift_up_blend_mask<S>());
}

template<int S>
KEWB_FORCE_INLINE __m512
shift_down_with_carry(__m512 lo, __m512 hi)
{
    return blend(rotate_down<S>(lo), rotate_down<S>(hi), shift_down_blend_mask<S>());
}

template<int S>
KEWB_FORCE_INLINE __m512
shift_up_and_fill(__m512 r0, float fill)
{
    return blend(rotate_up<S>(r0), load_value(fill), shift_up_blend_mask<S>());
}

template<int S>
KEWB_FORCE_INLINE __m512
shift_down_and_fill(__m512 r0, float fill)
{
    return blend(rotate_down<S>(r0), load_value(fill), shift_down_blend_mask<S>());
}

KEWB_FORCE_INLINE __m512
fused_multiply_add(__m512 r0, __m512 r1, __m512 acc)
{
    return _mm512_fmadd_ps(r0, r1, acc);
}


KEWB_FORCE_INLINE __m512
mask_permute(__m512 r0, __m512 r1, __m512i perm, uint32_t mask)
{
    return _mm512_mask_permutexvar_ps(r0, (__mmask16) mask, perm, r1);
}

KEWB_FORCE_INLINE __m512
mask_permute2(__m512 r0, __m512 r1, __m512i perm, uint32_t mask)
{
    return _mm512_mask_permutex2var_ps(r0, (__mmask16) mask, perm, r1);
}


KEWB_FORCE_INLINE __m512
minimum(__m512 r0, __m512 r1)
{
    return _mm512_min_ps(r0, r1);
}

__m512
KEWB_FORCE_INLINE maximum(__m512 r0, __m512 r1)
{
    return _mm512_max_ps(r0, r1);
}

KEWB_FORCE_INLINE __m512
compare_with_exchange(__m512 vals, __m512i perm, uint32_t mask)
{
    __m512  exch = permute(vals, perm);
    __m512  vmin = minimum(vals, exch);
    __m512  vmax = maximum(vals, exch);

    return blend(vmin, vmax, mask);
}

template<unsigned A=0, unsigned B=0, unsigned C=0, unsigned D=0,
         unsigned E=0, unsigned F=0, unsigned G=0, unsigned H=0,
         unsigned I=0, unsigned J=0, unsigned K=0, unsigned L=0,
         unsigned M=0, unsigned N=0, unsigned O=0, unsigned P=0>
KEWB_FORCE_INLINE constexpr uint32_t
make_bitmask()
{
    static_assert((A < 2) && (B < 2) && (C < 2) && (D < 2) &&
                  (E < 2) && (F < 2) && (G < 2) && (H < 2) &&
                  (I < 2) && (J < 2) && (K < 2) && (L < 2) &&
                  (M < 2) && (N < 2) && (O < 2) && (P < 2));

    return ((A <<  0) | (B <<  1) | (C <<  2) | (D <<  3) |
            (E <<  4) | (F <<  5) | (G <<  6) | (H <<  7) |
            (I <<  8) | (J <<  9) | (K << 10) | (L << 11) |
            (M << 12) | (N << 13) | (O << 14) | (P << 15));
}

KEWB_FORCE_INLINE __m512
sort_two_lanes_of_8(float_512 vals)
{
    //- Precompute the permutations and bitmasks for the 6 stages of this bitonic sorting sequence.
    //                                   0   1   2   3   4   5   6   7     0   1   2   3   4   5   6   7
    //                                  ---------------------------------------------------
    integer_512 const     perm0 = make_perm<1,  0,  3,  2,  5,  4,  7,  6,    9,  8, 11, 10, 13, 12, 15, 14>();
    constexpr m512  mask0 = make_bitmask<0,  1,  0,  1,  0,  1,  0,  1,    0,  1,  0,  1,  0,  1,  0,  1>();

    integer_512 const     perm1 = make_perm<3,  2,  1,  0,  7,  6,  5,  4,   11, 10,  9,  8, 15, 14, 13, 12>();
    constexpr m512  mask1 = make_bitmask<0,  0,  1,  1,  0,  0,  1,  1,    0,  0,  1,  1,  0,  0,  1,  1>();

    integer_512 const     perm2 = make_perm<1,  0,  3,  2,  5,  4,  7,  6,    9,  8, 11, 10, 13, 12, 15, 14>();
    constexpr m512  mask2 = make_bitmask<0,  1,  0,  1,  0,  1,  0,  1,    0,  1,  0,  1,  0,  1,  0,  1>();

    integer_512 const     perm3 = make_perm<7,  6,  5,  4,  3,  2,  1,  0,   15, 14, 13, 12, 11, 10,  9,  8>();
    constexpr m512  mask3 = make_bitmask<0,  0,  0,  0,  1,  1,  1,  1,    0,  0,  0,  0,  1,  1,  1,  1>();

    integer_512 const     perm4 = make_perm<2,  3,  0,  1,  6,  7,  4,  5,   10, 11,  8,  9, 14, 15, 12, 13>();
    constexpr m512  mask4 = make_bitmask<0,  0,  1,  1,  0,  0,  1,  1,    0,  0,  1,  1,  0,  0,  1,  1>();

    integer_512 const     perm5 = make_perm<1,  0,  3,  2,  5,  4,  7,  6,    9,  8, 11, 10, 13, 12, 15, 14>();
    constexpr m512  mask5 = make_bitmask<0,  1,  0,  1,  0,  1,  0,  1,    0,  1,  0,  1,  0,  1,  0,  1>();

    vals = compare_with_exchange(vals, perm0, mask0);
    vals = compare_with_exchange(vals, perm1, mask1);
    vals = compare_with_exchange(vals, perm2, mask2);
    vals = compare_with_exchange(vals, perm3, mask3);
    vals = compare_with_exchange(vals, perm4, mask4);
    vals = compare_with_exchange(vals, perm5, mask5);

    return vals;
}

KEWB_FORCE_INLINE __m512
sort_two_lanes_of_7(float_512 vals)
{
    //- Precompute the permutations and bitmasks for the 6 stages of this bitonic sorting sequence.
    //                                   0   1   2   3   4   5   6   7     0   1   2   3   4   5   6   7
    //                                  ---------------------------------------------------
    integer_512 const     perm0 = make_perm<4,  5,  6,  3,  0,  1,  2,  7,   12, 13, 14, 11,  8,  9, 10, 15>();
    constexpr m512  mask0 = make_bitmask<0,  0,  0,  0,  1,  1,  1,  0,    0,  0,  0,  0,  1,  1,  1,  0>();

    integer_512 const     perm1 = make_perm<2,  3,  0,  1,  6,  5,  4,  7,   10, 11,  8,  9, 14, 13, 12, 15>();
    constexpr m512  mask1 = make_bitmask<0,  0,  1,  1,  0,  0,  1,  0,   0,  0,  1,  1,  0,  0,  1,  0>();

    integer_512 const     perm2 = make_perm<1,  0,  4,  5,  2,  3,  6,  7,    9,  8, 12, 13, 10, 11, 14, 15>();
    constexpr m512  mask2 = make_bitmask<0,  1,  0,  0,  1,  1,  0,  0,    0,  1,  0,  0,  1,  1,  0,  0>();

    integer_512 const     perm3 = make_perm<0,  1,  3,  2,  5,  4,  6,  7,    8,  9, 11, 10, 13, 12, 14, 15>();
    constexpr m512  mask3 = make_bitmask<0,  0,  0,  1,  0,  1,  0,  0,    0,  0,  0,  1,  0,  1,  0,  0>();

    integer_512 const     perm4 = make_perm<0,  4,  2,  6,  1,  5,  3,  7,    8, 12, 10, 14,  9, 13, 11, 15>();
    constexpr m512  mask4 = make_bitmask<0,  0,  0,  0,  1,  0,  1,  0,    0,  0,  0,  0,  1,  0,  1,  0>();

    integer_512 const     perm5 = make_perm<0,  2,  1,  4,  3,  6,  5,  7,    8, 10,  9, 12, 11, 14, 13, 15>();
    constexpr m512  mask5 = make_bitmask<0,  0,  1,  0,  1,  0,  1,  0,    0,  0,  1,  0,  1,  0,  1,  0>();

    vals = compare_with_exchange(vals, perm0, mask0);
    vals = compare_with_exchange(vals, perm1, mask1);
    vals = compare_with_exchange(vals, perm2, mask2);
    vals = compare_with_exchange(vals, perm3, mask3);
    vals = compare_with_exchange(vals, perm4, mask4);
    vals = compare_with_exchange(vals, perm5, mask5);

    return vals;
}

}       //- simd namespace
#endif  //- KEWB_SIMD_HPP_DEFINED