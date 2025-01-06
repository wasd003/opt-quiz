#include "simd.hpp"
#include <quiz/base.h>

void vanilla_median_kernel(const std::vector<float>& src, std::vector<float>& answer)
{
    assert(src.size() - answer.size() == 6);
    std::array<float, 7>  seven;
    for (size_t i = 0;  i + 7 <= src.size();  i ++ ) {
        std::copy(src.begin() + i, src.begin() + i + 7, seven.begin());
        std::sort(seven.begin(), seven.end());
        answer[i] = seven[3];
    }
}

// [0,6] [1,7]
// [2,8] [3,9]
// [4,10] [5,11]
// [6,12] [7,13]
// [8,14] [9,15] 
// ...
// [14,20] [15,21]
void simd_median_kernel(const std::vector<float>& src, std::vector<float>& answer) {
    assert(src.size() - answer.size() == 6);
    const int n = src.size();
    simd::float_512 data, cur, next;
    UNUSED(data);
    UNUSED(cur);
    next = simd::load_from(src.data());
    int i = 0;
    for (i = 0; i + 32 < n; i += 16) {
        cur = next;
        next = simd::load_from(src.data() + i + 16);
        // simd::r512f vals = simd::sort_two_lanes_of_7(cur);
        // simd::store_to(answer.data() + i, vals);
    }

    /// fallback to vanilla way
    for (; i + 6 < n; i++) {
        std::array<float, 7> seven;
        std::copy(src.begin() + i, src.begin() + i + 7, seven.begin());
        std::sort(seven.begin(), seven.end());
        answer[i] = seven[3];
    }
}

void print(auto&& data, auto&& name) {
    std::cout << name;
    std::for_each(data.begin(), data.end(), [](auto x) { std::cout << x << " "; });
    std::cout << std::endl;
}

int main() {
#if 0
    constexpr static int N = 10;
    std::vector<float> src(N);
    std::vector<float> vanilla_answer(src.size() - 6);
    std::vector<float> algo_answer(src.size() - 6);
    std::srand(static_cast<unsigned int>(std::time(nullptr)));
    std::generate(src.begin(), src.end(), []() { return static_cast<float>(std::rand()) / RAND_MAX; });
    print(src, "src");

    vanilla_median_kernel(src, vanilla_answer);
    simd_median_kernel(src, algo_answer);

    print(vanilla_answer, "vanilla_answer");
    print(algo_answer, "algo_answer");
#endif


    std::vector<float> data0(16), data1(16);
    std::iota(data0.begin(), data0.end(), 0.0f);
    std::iota(data1.begin(), data1.end(), 16.0f);
    auto lo = simd::load_from(data0.data());
    auto hi = simd::load_from(data1.data());
    PRINT_REG(lo);
    PRINT_REG(hi);
    simd::inplace_shift_lo_with_carry<2>(lo, hi);
    PRINT_REG(lo);
    PRINT_REG(hi);


    return 0;
}