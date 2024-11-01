#include <immintrin.h>
#include <quiz/base.h>
#include <x86intrin.h>

using namespace std;

constexpr static int N = 2048;

alignas(64) double A[N][N];
alignas(64) double B[N][N];
alignas(64) double C[N][N];
alignas(64) double ans[N][N];

void init() {
    for (int i = 0; i < N; i ++ )
        for (int j = 0; j < N; j ++ ) {
            A[i][j] = random(1, 1);
            B[i][j] = random(1, 1);
            ans[i][j] = C[i][j] = 0;
        }
}

void mat_vanilla() {
    for (int i = 0; i < N; i ++ ) {
        for (int j = 0; j < N; j ++ ) {
            for (int k = 0; k < N; k ++ ) {
                ans[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

// subword parallelism
void mat_subword_parallsim() {
    for (int i = 0; i < N; i ++ ) {
        int j = 0;
        for (; j + 8 <= N; j += 8) {
            auto c = _mm512_setzero_pd();
            for (int k = 0; k < N; k ++ ) {
                auto a = _mm512_set1_pd(A[i][k]);
                auto b = _mm512_loadu_pd(&B[k][j]);
                c = _mm512_fmadd_pd(a, b, c);
            }
            _mm512_storeu_pd(&C[i][j], c);
        }
        for (; j < N; j ++ ) {
            for (int k = 0; k < N; k ++ ) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

// subword parallelism +
// instruction parallelism
void mat_subword_instruction_parallsim() {
    constexpr static int unroll = 4;
    constexpr static int stride = unroll * 8;
    __m512d c[unroll];
    for (int i = 0; i < N; i ++ ) {
        int j = 0;
        for (; j + stride <= N; j += stride) {
            for (int k = 0; k < unroll; k ++ ) {
                c[k] = _mm512_setzero_pd();
            }
            for (int k = 0; k < N; k ++ ) {
                auto a = _mm512_set1_pd(A[i][k]);
                for (int l = 0; l < unroll; l ++ ) {
                    auto b = _mm512_loadu_pd(&B[k][j + l * 8]);
                    c[l] = _mm512_fmadd_pd(a, b, c[l]);
                }
            }
            for (int k = 0; k < unroll; k ++ ) {
                _mm512_storeu_pd(&C[i][j + k * 8], c[k]);
            }
        }
        for (; j < N; j ++ ) {
            for (int k = 0; k < N; k ++ ) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

void correct_test() {
    mat_vanilla();
    mat_subword_instruction_parallsim();
    for (int i = 0; i < N; i ++ )
        for (int j = 0; j < N; j ++ ) {
            if (fabs(C[i][j] - ans[i][j]) > 1e-6) {
                printf("ERROR: C[%d][%d] = %lf, ans[%d][%d] = %lf\n", i, j, C[i][j], i, j, ans[i][j]);
                return;
            }

        }
    cout << "Correct" << endl;
}

void perf_test() {
    ClockWatch<CLOCK_REALTIME> clock;
    mat_subword_instruction_parallsim();
    auto tv = clock.Get();
    std::cout << tv << std::endl;
}

int main() {
    init();
    perf_test();
    return 0;
}