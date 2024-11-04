#include <cstdlib>
#include <iostream>
#include <quiz/base.h>
#include <benchmark/benchmark.h>

using LL = long long;

constexpr auto N = (1 << 22);

static void BM_branch(benchmark::State& state) {
    srand(1);
    std::vector<LL> v1(N), v2(N);
    std::vector<int> c1(N);
    for (int i = 0; i < N; i ++ ) {
        v1[i] = rand();
        v2[i] = rand();
        c1[i] = rand() & 1;
    }
    LL *p1 = v1.data();
    LL *p2 = v2.data();
    int *b1 = c1.data();

    for (auto _ : state) {
        LL a1 = 0, a2 = 0;
        for (int i = 0; i < N; i ++ ) {
            if (b1[i]) {
                a1 += p1[i] - p2[i];
            } else {
                a2 += p1[i] * p2[i];
            }
        }
        benchmark::DoNotOptimize(a1);
        benchmark::DoNotOptimize(a2);
        benchmark::ClobberMemory();
    }
    state.SetBytesProcessed(N * state.iterations() * sizeof(LL));
}
BENCHMARK(BM_branch);

BENCHMARK_MAIN();