#include "simd.hpp"
#include <quiz/base.h>
#include <benchmark/benchmark.h>

template<int KernelSize>
void vanilla_conv_kernel(const auto& src, const auto& kernel, auto& answer) {
    assert(kernel.size() == KernelSize);
    assert(src.size() - kernel.size() + 1 == answer.size());
    for (size_t i = 0; i < answer.size(); i++) {
        float sum = 0;
        for (size_t j = 0; j < kernel.size(); j++) {
            sum += src[i + j] * kernel[j];
        }
        answer[i] = sum;
    }
}

template<bool Aligned, int KernelSize>
void simd_conv_kernel(const auto& src, const auto& kernel, auto& answer) {
    assert(kernel.size() == KernelSize);
    assert(src.size() - kernel.size() + 1 == answer.size());
    static_assert(KernelSize <= 16 && KernelSize > 0);
    size_t i;
    auto next = simd::load_from<Aligned>(src.data());
    simd::float_512 conv_kernel[KernelSize];
    for (int j = 0; j < KernelSize; j ++ ) {
        conv_kernel[j] = simd::load_value(kernel[j]);
    }
    for (i = 0; i + 32 < src.size(); i += 16) {
        auto lo = next;
        next = simd::load_from<Aligned>(src.data() + i + 16);
        auto hi = next;
        auto data = simd::load_value(0.0f);
        for (int j = 0; j < KernelSize; j ++ ) {
            data = simd::fused_multiply_add(conv_kernel[j], lo, data);
            simd::inplace_shift_lo_with_carry<1>(lo, hi);
        }
        simd::store_to<Aligned>(answer.data() + i, data);
    }

    for (; i < answer.size(); i ++ ) {
        float sum = 0;
        for (size_t j = 0; j < kernel.size(); j++) {
            sum += src[i + j] * kernel[j];
        }
        answer[i] = sum;
    }
}

static auto create_input_vector(int N, int KernelSize) {
    AlignedVector<float, 64> src(N);
    AlignedVector<float, 64> kernel(KernelSize);
    AlignedVector<float, 64> vanilla_answer(src.size() - KernelSize + 1);
    AlignedVector<float, 64> simd_answer(src.size() - KernelSize + 1);

    std::srand(static_cast<unsigned int>(std::time(nullptr)));
    std::generate(src.begin(), src.end(), []() { return static_cast<float>(std::rand()) / RAND_MAX; });
    std::generate(kernel.begin(), kernel.end(), []() { return static_cast<float>(std::rand()) / RAND_MAX; });

    return std::make_tuple(src, kernel, vanilla_answer, simd_answer);
}

[[maybe_unused]] static void correct_test() {
    constexpr static int N = 1e7;
    constexpr static int KernelSize = 16;

    auto [src, kernel, vanilla_answer, simd_answer] = create_input_vector(N, KernelSize);

    vanilla_conv_kernel<KernelSize>(src, kernel, vanilla_answer);
    simd_conv_kernel<true, KernelSize>(src, kernel, simd_answer);

    // PRINT_VEC(src);
    // PRINT_VEC(kernel);
    // PRINT_VEC(vanilla_answer);
    // PRINT_VEC(simd_answer);

    equal_vec(vanilla_answer, simd_answer);
}

constexpr static int KernelSize = 3;

static void bm_vanilla_kernel(benchmark::State& state) {
    const int N = state.range(0);
    auto [src, kernel, vanilla_answer, simd_answer] = create_input_vector(N, KernelSize);

    for (auto _ : state) {
        vanilla_conv_kernel<KernelSize>(src, kernel, vanilla_answer);
        benchmark::DoNotOptimize(vanilla_answer);
        benchmark::ClobberMemory();
    }
    state.SetBytesProcessed(N * state.iterations());
}

static void bm_simd_kernel(benchmark::State& state) {
    const int N = state.range(0);
    auto [src, kernel, vanilla_answer, simd_answer] = create_input_vector(N, KernelSize);

    for (auto _ : state) {
        simd_conv_kernel<false, KernelSize>(src, kernel, simd_answer);
        benchmark::DoNotOptimize(simd_answer);
        benchmark::ClobberMemory();
    }
    state.SetBytesProcessed(N * state.iterations());
}

static void bm_aligned_simd_kernel(benchmark::State& state) {
    const int N = state.range(0);
    auto [src, kernel, vanilla_answer, simd_answer] = create_input_vector(N, KernelSize);

    for (auto _ : state) {
        simd_conv_kernel<true, KernelSize>(src, kernel, simd_answer);
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
//     return 0;
// }