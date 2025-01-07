#include "simd.hpp"
#include <quiz/base.h>
#include <benchmark/benchmark.h>

static void vanilla_median_kernel(const auto& src, auto& answer)
{
    assert(src.size() - answer.size() == 6);
    std::array<float, 7>  seven;
    for (size_t i = 0;  i + 7 <= src.size();  i ++ ) {
        std::copy(src.begin() + i, src.begin() + i + 7, seven.begin());
        std::sort(seven.begin(), seven.end());
        answer[i] = seven[3];
    }
}

template<bool Aligned>
static void simd_median_kernel(const auto& src, auto& answer) {
    assert(src.size() - answer.size() == 6);
    const int n = src.size();
    auto data = simd::load_value(0.0f);
    auto next = simd::load_from<Aligned>(src.data());
    const auto load_perm = simd::make_perm<0, 1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7, 8>();
    const auto store_perm = simd::make_perm<3, 11, 3, 11, 3, 11, 3, 11, 3, 11, 3, 11, 3, 11, 3, 11>();
    int i = 0;
    for (i = 0; i + 32 < n; i += 16) {
        auto lo = next;
        next = simd::load_from<Aligned>(src.data() + i + 16);
        auto hi = next;
        for (int j = 0; j < 16; j += 2) {
            auto tmp = simd::permute(lo, load_perm);
            tmp = simd::sort_two_lanes_of_7(tmp);
            data = simd::masked_permute(tmp, data, store_perm, 3 << j);
            simd::inplace_shift_lo_with_carry<2>(lo, hi);
        }
        simd::store_to<Aligned>(answer.data() + i, data);
    }

    /// fallback to vanilla way
    for (; i + 6 < n; i++) {
        std::array<float, 7> seven;
        std::copy(src.begin() + i, src.begin() + i + 7, seven.begin());
        std::sort(seven.begin(), seven.end());
        answer[i] = seven[3];
    }
}

[[maybe_unused]] static void correct_test() {
    constexpr static int N = 1e7;
    std::vector<float> src(N);
    std::vector<float> vanilla_answer(src.size() - 6);
    std::vector<float> simd_answer(src.size() - 6);
#if 1
    std::srand(static_cast<unsigned int>(std::time(nullptr)));
    std::generate(src.begin(), src.end(), []() { return static_cast<float>(std::rand()) / RAND_MAX; });
#else
    for (int i = 0; i < N; i ++ ) src[i] = i;
    std::shuffle(src.begin(), src.end(), std::mt19937(std::random_device()()));
#endif

    vanilla_median_kernel(src, vanilla_answer);
    simd_median_kernel<false>(src, simd_answer);

    equal_vec(vanilla_answer, simd_answer);
}


static auto create_input_vector(int N) {
    AlignedVector<float, 64> src(N);
    AlignedVector<float, 64> vanilla_answer(src.size() - 6);
    AlignedVector<float, 64> simd_answer(src.size() - 6);
    std::srand(static_cast<unsigned int>(std::time(nullptr)));
    std::generate(src.begin(), src.end(), []() { return static_cast<float>(std::rand()) / RAND_MAX; });
    return std::make_tuple(src, vanilla_answer, simd_answer);
}

static void bm_vanilla_kernel(benchmark::State& state) {
    const int N = state.range(0);
    auto [src, vanilla_answer, simd_answer] = create_input_vector(N);

    for (auto _ : state) {
        vanilla_median_kernel(src, vanilla_answer);
        benchmark::DoNotOptimize(vanilla_answer);
        benchmark::ClobberMemory();
    }
    state.SetBytesProcessed(N * state.iterations());
}

static void bm_simd_kernel(benchmark::State& state) {
    const int N = state.range(0);
    auto [src, vanilla_answer, simd_answer] = create_input_vector(N);

    for (auto _ : state) {
        simd_median_kernel<false>(src, simd_answer);
        benchmark::DoNotOptimize(simd_answer);
        benchmark::ClobberMemory();
    }
    state.SetBytesProcessed(N * state.iterations());
}

static void bm_aligned_simd_kernel(benchmark::State& state) {
    const int N = state.range(0);
    auto [src, vanilla_answer, simd_answer] = create_input_vector(N);

    for (auto _ : state) {
        simd_median_kernel<true>(src, simd_answer);
        benchmark::DoNotOptimize(simd_answer);
        benchmark::ClobberMemory();
    }
    state.SetBytesProcessed(N * state.iterations());
}

constexpr static int MIN_ELEMENT_NR = 1e6;
constexpr static int MAX_ELEMENT_NR = 1e7;

BENCHMARK(bm_vanilla_kernel) ->Range(MIN_ELEMENT_NR, MAX_ELEMENT_NR);
BENCHMARK(bm_simd_kernel) ->Range(MIN_ELEMENT_NR, MAX_ELEMENT_NR);
BENCHMARK(bm_aligned_simd_kernel) ->Range(MIN_ELEMENT_NR, MAX_ELEMENT_NR);

BENCHMARK_MAIN();

// int main() {
//     correct_test();
// }