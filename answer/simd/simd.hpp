#pragma once

#include <immintrin.h>
#include <cstdint>
#include <cstdio>
#include <tuple>
#include <type_traits>

class simd {
public:
    using double_128 = __m128d;
    using float_128 = __m128;
    using integer_128 = __m128i;

    using double_256 = __m256d;
    using float_256 = __m256;
    using integer_256 = __m256i;

    using double_512 = __m512d;
    using float_512 = __m512;
    using integer_512 = __m512i;

private:
    #define force_inline __attribute__((__always_inline__))

    template <int R>
    static force_inline auto consteval rotate_offset() {
        constexpr int S = (R > 0) ? (16 - (R % 16)) : -R;
        constexpr int A = (S + 0) % 16;
        constexpr int B = (S + 1) % 16;
        constexpr int C = (S + 2) % 16;
        constexpr int D = (S + 3) % 16;
        constexpr int E = (S + 4) % 16;
        constexpr int F = (S + 5) % 16;
        constexpr int G = (S + 6) % 16;
        constexpr int H = (S + 7) % 16;
        constexpr int I = (S + 8) % 16;
        constexpr int J = (S + 9) % 16;
        constexpr int K = (S + 10) % 16;
        constexpr int L = (S + 11) % 16;
        constexpr int M = (S + 12) % 16;
        constexpr int N = (S + 13) % 16;
        constexpr int O = (S + 14) % 16;
        constexpr int P = (S + 15) % 16;
        return std::make_tuple(A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P);
    }

    template <int R>
    static force_inline float_512 rotate(float_512 r0) {
        if constexpr ((R % 16) == 0) {
            return r0;
        } else {
            const auto [A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P] =
                rotate_offset<R>();
            return _mm512_permutexvar_ps(
                _mm512_setr_epi32(A, B, C, D, E, F, G, H, I, J, K, L, M, N, O,
                                  P),
                r0);
        }
    }

    template <int R>
    static force_inline integer_512 rotate(integer_512 r0) {
        if constexpr ((R % 16) == 0) {
            return r0;
        } else {
            const auto [A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P] =
                rotate_offset<R>();
            return _mm512_permutexvar_epi32(
                _mm512_setr_epi32(A, B, C, D, E, F, G, H, I, J, K, L, M, N, O,
                                  P),
                r0);
        }
    }

    template <int BIAS, uint32_t MASK>
    static force_inline integer_512 make_shift_permutation() {
        constexpr int32_t a = ((BIAS + 0) % 16) | ((MASK & 1u) ? 0x10 : 0);
        constexpr int32_t b =
            ((BIAS + 1) % 16) | ((MASK & 1u << 1u) ? 0x10 : 0);
        constexpr int32_t c =
            ((BIAS + 2) % 16) | ((MASK & 1u << 2u) ? 0x10 : 0);
        constexpr int32_t d =
            ((BIAS + 3) % 16) | ((MASK & 1u << 3u) ? 0x10 : 0);
        constexpr int32_t e =
            ((BIAS + 4) % 16) | ((MASK & 1u << 4u) ? 0x10 : 0);
        constexpr int32_t f =
            ((BIAS + 5) % 16) | ((MASK & 1u << 5u) ? 0x10 : 0);
        constexpr int32_t g =
            ((BIAS + 6) % 16) | ((MASK & 1u << 6u) ? 0x10 : 0);
        constexpr int32_t h =
            ((BIAS + 7) % 16) | ((MASK & 1u << 7u) ? 0x10 : 0);
        constexpr int32_t i =
            ((BIAS + 8) % 16) | ((MASK & 1u << 8u) ? 0x10 : 0);
        constexpr int32_t j =
            ((BIAS + 9) % 16) | ((MASK & 1u << 9u) ? 0x10 : 0);
        constexpr int32_t k =
            ((BIAS + 10) % 16) | ((MASK & 1u << 10u) ? 0x10 : 0);
        constexpr int32_t l =
            ((BIAS + 11) % 16) | ((MASK & 1u << 11u) ? 0x10 : 0);
        constexpr int32_t m =
            ((BIAS + 12) % 16) | ((MASK & 1u << 12u) ? 0x10 : 0);
        constexpr int32_t n =
            ((BIAS + 13) % 16) | ((MASK & 1u << 13u) ? 0x10 : 0);
        constexpr int32_t o =
            ((BIAS + 14) % 16) | ((MASK & 1u << 14u) ? 0x10 : 0);
        constexpr int32_t p =
            ((BIAS + 15) % 16) | ((MASK & 1u << 15u) ? 0x10 : 0);

        return _mm512_setr_epi32(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o,
                                 p);
    }

    template <int S>
    static force_inline constexpr uint32_t shift_hi_blend_mask() {
        static_assert(S >= 0 && S <= 16);
        return (0xFFFFu << (unsigned)S) & 0xFFFFu;
    }

    template <int S>
    static force_inline constexpr uint32_t shift_lo_blend_mask() {
        static_assert(S >= 0 && S <= 16);
        return (0xFFFFu << (unsigned)(16 - S)) & 0xFFFFu;
    }

    force_inline float_512 static compare_with_exchange(float_512 vals,
                                                        integer_512 perm,
                                                        uint32_t mask) {
        float_512 exch = permute(vals, perm);
        float_512 vmin = minimum(vals, exch);
        float_512 vmax = maximum(vals, exch);

        return blend(vmin, vmax, mask);
    }

    static void print_vals(char const* pname, double const* pvals, int n) {
        if (pname != nullptr) {
            printf("reg %s:\n", pname);
        }

        for (int i = 0; i < n; ++i) {
            printf("%6.1f", pvals[i]);
        }
        printf("\n");
        fflush(stdout);
    }

    static void print_vals(char const* pname, float const* pvals, int n) {
        if (pname != nullptr) {
            printf("reg %s:\n", pname);
        }

        for (int i = 0; i < n; ++i) {
            printf("%6.1f", pvals[i]);
        }
        printf("\n");
        fflush(stdout);
    }

    static void print_vals(char const* pname, int32_t const* pvals, int n) {
        if (pname != nullptr) {
            printf("reg %s:\n", pname);
        }

        for (int i = 0; i < n; ++i) {
            printf("%6d", pvals[i]);
        }
        printf("\n");
        fflush(stdout);
    }

public:
    /**
     * SEQ: 0
     * @load_value: load a immediate value into SIMD register
     * @load_values: load a set of immediate values into SIMD register
     */
    static force_inline float_512 load_value(float immediate_val) {
        return _mm512_set1_ps(immediate_val);
    }

    static force_inline integer_512 load_value(int32_t immediate_val) {
        return _mm512_set1_epi32(immediate_val);
    }

    template <int A, int B, int C, int D, int E, int F, int G, int H, int I,
              int J, int K, int L, int M, int N, int O, int P>
    static force_inline integer_512 load_values() {
        return _mm512_setr_epi32(A, B, C, D, E, F, G, H, I, J, K, L, M, N, O,
                                 P);
    }

    static force_inline integer_512 load_values(int a, int b, int c, int d,
                                                int e, int f, int g, int h,
                                                int i, int j, int k, int l,
                                                int m, int n, int o, int p) {
        return _mm512_setr_epi32(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o,
                                 p);
    }

    static force_inline float_512 load_values(float a, float b, float c,
                                              float d, float e, float f,
                                              float g, float h, float i,
                                              float j, float k, float l,
                                              float m, float n, float o,
                                              float p) {
        return _mm512_setr_ps(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p);
    }

    /**
     * SEQ: 1
     * @load_from: load data from memory into SIMD register
     */
    static force_inline double_512 load_from(const double* psrc) {
        return _mm512_loadu_pd(psrc);
    }

    static force_inline float_512 load_from(const float* psrc) {
        return _mm512_loadu_ps(psrc);
    }

    static force_inline integer_512 load_from(const int32_t* psrc) {
        return _mm512_loadu_epi32(psrc);
    }

    /**
     * SEQ: 2
     * @masked_load_from: load data from [memory] or [SIMD register/immediate value]
     *      - if mask bit is 1, load from memory
     *      - if mask bit is 0, load from SIMD register/immediate value
     */
    static force_inline float_512 masked_load_from(const float* psrc,
                                                   float_512 simd_reg,
                                                   uint32_t mask) {
        return _mm512_mask_loadu_ps(simd_reg, (__mmask16)mask, psrc);
    }

    static force_inline float_512 masked_load_from(const float* psrc,
                                                   float immediate_val,
                                                   uint32_t mask) {
        return _mm512_mask_loadu_ps(_mm512_set1_ps(immediate_val),
                                    (__mmask16)mask, psrc);
    }

    /**
     * SEQ: 3
     * @store_to: store data from SIMD register to memory
     */
    static force_inline void store_to(void* pdst, integer_512 r) {
        _mm512_mask_storeu_epi32(pdst, (__mmask16)0xFFFFu, r);
    }

    static force_inline void store_to(void* pdst, float_512 r) {
        _mm512_mask_storeu_ps(pdst, (__mmask16)0xFFFFu, r);
    }

    /**
     * SEQ: 4
     * @masked_store_to: store data to memory based on mask
     *    - if mask bit is 1, store from SIMD register
     *    - if mask bit is 0, leave memory unchanged
     */
    static force_inline void masked_store_to(void* pdst, float_512 r,
                                             uint32_t mask) {
        _mm512_mask_storeu_ps(pdst, (__mmask16)mask, r);
    }

    /**
     * SEQ: 5
     * @blend: blend data from two SIMD registers based on mask
     *    - if mask bit is 1, use data from r1
     *    - if mask bit is 0, use data from r0
     */
    static force_inline float_512 blend(float_512 r0, float_512 r1,
                                        uint32_t mask) {
        return _mm512_mask_blend_ps((__mmask16)mask, r0, r1);
    }

    /**
     * SEQ: 6
     * @permute: export answer register based on r and perm
     *      answer[i] = reg[perm[i]]
     */
    static force_inline float_512 permute(float_512 reg, integer_512 perm) {
        return _mm512_permutexvar_ps(perm, reg);
    }

    static force_inline integer_512 permute(integer_512 reg, integer_512 perm) {
        return _mm512_permutexvar_epi32(perm, reg);
    }

    template <int A, int B, int C, int D, int E, int F, int G, int H, int I,
              int J, int K, int L, int M, int N, int O, int P>
    static force_inline integer_512 permute(integer_512 reg) {
        return _mm512_permutexvar_epi32(
            load_values<A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P>(), reg);
    }

    /**
     * SEQ: 7
     * @masked_permute: export answer register based on r and perm and mask
     *     - if mask[i] == 1, then answer[i] = reg[perm[i]]
     *     - if mask[i] == 0, then answer[i] = other_reg[i]
     */
    static force_inline float_512 masked_permute(float_512 reg,
                                                 float_512 other_reg,
                                                 integer_512 perm,
                                                 uint32_t mask) {
        return _mm512_mask_permutexvar_ps(other_reg, (__mmask16)mask, perm,
                                          reg);
    }

    /**
     * SEQ: 8
     * @make_perm: create a permutation register based on immediate values
     * @make_bitmask: create a uint32_t bitmask
     */
    template <unsigned... IDXS>
    static force_inline auto make_perm() {
        static_assert(sizeof...(IDXS) == 8 || sizeof...(IDXS) == 16);

        if constexpr (sizeof...(IDXS) == 8) {
            return _mm256_setr_epi32(IDXS...);
        } else {
            return load_values<IDXS...>();
        }
    }

    template <unsigned A = 0, unsigned B = 0, unsigned C = 0, unsigned D = 0,
              unsigned E = 0, unsigned F = 0, unsigned G = 0, unsigned H = 0,
              unsigned I = 0, unsigned J = 0, unsigned K = 0, unsigned L = 0,
              unsigned M = 0, unsigned N = 0, unsigned O = 0, unsigned P = 0>
    static force_inline constexpr uint32_t make_bitmask() {
        static_assert((A < 2) && (B < 2) && (C < 2) && (D < 2) && (E < 2) &&
                      (F < 2) && (G < 2) && (H < 2) && (I < 2) && (J < 2) &&
                      (K < 2) && (L < 2) && (M < 2) && (N < 2) && (O < 2) &&
                      (P < 2));

        return ((A << 0) | (B << 1) | (C << 2) | (D << 3) | (E << 4) |
                (F << 5) | (G << 6) | (H << 7) | (I << 8) | (J << 9) |
                (K << 10) | (L << 11) | (M << 12) | (N << 13) | (O << 14) |
                (P << 15));
    }

    /**
     * SEQ: 9
     * @rotate_lo: rotate simd register towards lower direction
     * @rotate_hi: rotate simd register towards higher direction
     */
    template <int R>
    static force_inline float_512 rotate_lo(float_512 r0) {
        static_assert(R >= 0);
        return rotate<-R>(r0);
    }

    template <int R>
    static force_inline float_512 rotate_hi(float_512 r0) {
        static_assert(R >= 0);
        return rotate<R>(r0);
    }

    template <int R>
    static force_inline integer_512 rotate_lo(integer_512 r0) {
        static_assert(R >= 0);
        return rotate<-R>(r0);
    }

    template <int R>
    static force_inline integer_512 rotate_hi(integer_512 r0) {
        static_assert(R >= 0);
        return rotate<R>(r0);
    }

    /**
     * SEQ: 10
     * @shift_lo: shift simd register towards lower direction
     * @shift_hi: shift simd register towards higher direction
     *      - discard the values that are shifted out
     *      - leave the empty slots with 0
     */
    template <int S>
    static force_inline float_512 shift_lo(float_512 r0) {
        static_assert(S >= 0 && S <= 16);
        return blend(rotate_lo<S>(r0), load_value(static_cast<float>(0)),
                     shift_lo_blend_mask<S>());
    }

    template <int S>
    static force_inline float_512 shift_hi(float_512 r0) {
        static_assert(S >= 0 && S <= 16);
        return blend(load_value(static_cast<float>(0)), rotate_hi<S>(r0),
                     shift_hi_blend_mask<S>());
    }

    /**
     * SEQ: 11
     * @shift_lo_with_carry: suppose we have a ruler x points to lo.
     * now shift [lo][hi] towards lo, what falls into X will the final answer
     *    [  lo  ][  hi  ] <-
     *    |  X   |
     *
     * @shift_hi_with_carry: suppose we have a ruler x points to hi.
     * now shift [lo][hi] towards hi, what falls into X will the final answer
     *    ->[  lo  ][  hi  ]
     *              |  X   |
     */
    template <int S>
    static force_inline float_512 shift_lo_with_carry(float_512 lo,
                                                      float_512 hi) {
        static_assert(S >= 0 && S <= 16);
        return blend(rotate_lo<S>(lo), rotate_lo<S>(hi),
                     shift_lo_blend_mask<S>());
    }

    template <int S>
    static force_inline float_512 shift_hi_with_carry(float_512 lo,
                                                      float_512 hi) {
        static_assert(S >= 0 && S <= 16);
        return blend(rotate_hi<S>(lo), rotate_hi<S>(hi),
                     shift_hi_blend_mask<S>());
    }

    /**
     * SEQ: 12
     * @in_place_shift_lo_with_carry: same with shift_lo_with_carry, but in place
     * suppose: lo=[1,2,3,4], hi=[5,6,7,8]
     * shift_lo with 2. now lo=[3,4,5,6], hi=[7,8,0,0]
     */
    template <int S>
    static force_inline void inplace_shift_lo_with_carry(float_512& lo,
                                                         float_512& hi) {
        static_assert(S >= 0 && S <= 16);

        constexpr uint32_t zmask = (0xFFFFu >> (unsigned)S);
        constexpr uint32_t bmask = ~zmask & 0xFFFFu;
        integer_512 perm = make_shift_permutation<S, bmask>();

        lo = _mm512_permutex2var_ps(lo, perm, hi);
        hi = _mm512_maskz_permutex2var_ps((__mmask16)zmask, hi, perm, hi);
    }

    /**
     * SEQ: 13
     * @fused_multiply_add: answer = r0 * r1 + acc
     * @minimum: return the minimum value of r0 and r1 elementwise
     * @maximum: return the maxmum value of r0 and r1 elementwise
     */
    static force_inline float_512 fused_multiply_add(float_512 r0, float_512 r1,
                                                     float_512 acc) {
        return _mm512_fmadd_ps(r0, r1, acc);
    }

    static force_inline float_512 minimum(float_512 r0, float_512 r1) {
        return _mm512_min_ps(r0, r1);
    }

    static force_inline float_512 maximum(float_512 r0, float_512 r1) {
        return _mm512_max_ps(r0, r1);
    }

    /**
     * SEQ: 14
     * @sort_two_lanes_of_8: sort two lanes of 8 elements
     * @sort_two_lanes_of_7: sort two lanes of 7 elements
     */
    static force_inline float_512 sort_two_lanes_of_8(float_512 vals) {
        //- Precompute the permutations and bitmasks for the 6 stages of this bitonic sorting sequence.
        //                                   0   1   2   3   4   5   6   7     0   1   2   3   4   5   6   7
        //                                  ---------------------------------------------------
        integer_512 const perm0 =
            make_perm<1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14>();
        constexpr uint32_t mask0 =
            make_bitmask<0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1>();

        integer_512 const perm1 =
            make_perm<3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12>();
        constexpr uint32_t mask1 =
            make_bitmask<0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1>();

        integer_512 const perm2 =
            make_perm<1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14>();
        constexpr uint32_t mask2 =
            make_bitmask<0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1>();

        integer_512 const perm3 =
            make_perm<7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8>();
        constexpr uint32_t mask3 =
            make_bitmask<0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1>();

        integer_512 const perm4 =
            make_perm<2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13>();
        constexpr uint32_t mask4 =
            make_bitmask<0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1>();

        integer_512 const perm5 =
            make_perm<1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14>();
        constexpr uint32_t mask5 =
            make_bitmask<0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1>();

        vals = compare_with_exchange(vals, perm0, mask0);
        vals = compare_with_exchange(vals, perm1, mask1);
        vals = compare_with_exchange(vals, perm2, mask2);
        vals = compare_with_exchange(vals, perm3, mask3);
        vals = compare_with_exchange(vals, perm4, mask4);
        vals = compare_with_exchange(vals, perm5, mask5);

        return vals;
    }

    static force_inline float_512 sort_two_lanes_of_7(float_512 vals) {
        //- Precompute the permutations and bitmasks for the 6 stages of this bitonic sorting sequence.
        //                                   0   1   2   3   4   5   6   7     0   1   2   3   4   5   6   7
        //                                  ---------------------------------------------------
        integer_512 const perm0 =
            make_perm<4, 5, 6, 3, 0, 1, 2, 7, 12, 13, 14, 11, 8, 9, 10, 15>();
        constexpr uint32_t mask0 =
            make_bitmask<0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0>();

        integer_512 const perm1 =
            make_perm<2, 3, 0, 1, 6, 5, 4, 7, 10, 11, 8, 9, 14, 13, 12, 15>();
        constexpr uint32_t mask1 =
            make_bitmask<0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0>();

        integer_512 const perm2 =
            make_perm<1, 0, 4, 5, 2, 3, 6, 7, 9, 8, 12, 13, 10, 11, 14, 15>();
        constexpr uint32_t mask2 =
            make_bitmask<0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0>();

        integer_512 const perm3 =
            make_perm<0, 1, 3, 2, 5, 4, 6, 7, 8, 9, 11, 10, 13, 12, 14, 15>();
        constexpr uint32_t mask3 =
            make_bitmask<0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0>();

        integer_512 const perm4 =
            make_perm<0, 4, 2, 6, 1, 5, 3, 7, 8, 12, 10, 14, 9, 13, 11, 15>();
        constexpr uint32_t mask4 =
            make_bitmask<0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0>();

        integer_512 const perm5 =
            make_perm<0, 2, 1, 4, 3, 6, 5, 7, 8, 10, 9, 12, 11, 14, 13, 15>();
        constexpr uint32_t mask5 =
            make_bitmask<0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0>();

        vals = compare_with_exchange(vals, perm0, mask0);
        vals = compare_with_exchange(vals, perm1, mask1);
        vals = compare_with_exchange(vals, perm2, mask2);
        vals = compare_with_exchange(vals, perm3, mask3);
        vals = compare_with_exchange(vals, perm4, mask4);
        vals = compare_with_exchange(vals, perm5, mask5);

        return vals;
    }

    /**
     * SEQ: 15
     * debug print interface
     */
    #define PRINT_REG(R) simd::print_reg(#R, R)
    #define PRINT_MASK(M) simd::print_mask(#M, M, 16)
    #define PRINT_MASK8(M) simd::print_mask(#M, M, 8)
    #define PRINT_MASK16(M) simd::print_mask(#M, M, 16)
    #define PRINT_LINE() simd::printf("\n");

    static void print_reg(char const* pname, double_128 r) {
        double vals[2];

        _mm_storeu_pd(&vals[0], r);
        print_vals(pname, vals, 2);
    }

    static void print_reg(char const* pname, float_128 r) {
        float vals[4];

        _mm_storeu_ps(&vals[0], r);
        print_vals(pname, vals, 4);
    }

    static void print_reg(char const* pname, integer_128 r) {
        int32_t vals[4];

        _mm_storeu_epi32(&vals[0], r);
        print_vals(pname, vals, 4);
    }

    static void print_reg(char const* pname, double_256 r) {
        double vals[4];

        _mm256_storeu_pd(&vals[0], r);
        print_vals(pname, vals, 4);
    }

    static void print_reg(char const* pname, float_256 r) {
        float vals[8];

        _mm256_storeu_ps(&vals[0], r);
        print_vals(pname, vals, 8);
    }

    static void print_reg(char const* pname, integer_256 r) {
        int32_t vals[8];

        _mm256_storeu_si256((integer_256*)&vals[0], r);
        print_vals(pname, vals, 8);
    }

    static void print_reg(char const* pname, double_512 r) {
        double vals[8];

        _mm512_storeu_pd(&vals[0], r);
        print_vals(pname, vals, 8);
    }

    static void print_reg(char const* pname, float_512 r) {
        float vals[16];

        _mm512_storeu_ps(&vals[0], r);
        print_vals(pname, vals, 16);
    }

    static void print_reg(char const* pname, integer_512 r) {
        int32_t vals[16];

        _mm512_storeu_epi32(&vals[0], r);
        print_vals(pname, vals, 16);
    }

    static void print_mask(char const* pname, uint32_t mask, int bits) {
        if (pname != nullptr) {
            printf("mask %s:\n", pname);
        }

        uint32_t probe = 1;

        for (int i = 0; i < bits; ++i, probe <<= 1) {
            printf("%6d", (mask & probe) ? 1 : 0);
        }
        printf("\n");
        fflush(stdout);
    }

    static void print_mask(char const* pname, __m256i mask, int) {
        if (pname != nullptr) {
            printf("mask %s:\n", pname);
        }
        print_reg(nullptr, mask);
    }
};