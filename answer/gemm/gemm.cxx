#include <cassert>
#include <immintrin.h>
#include <quiz/base.h>
#include <x86intrin.h>

using namespace std;

constexpr static int N = 512;

alignas(64) double A[N][N];
alignas(64) double B[N][N];
alignas(64) double C[N][N];
alignas(64) double ans[N][N];

void init() {
    for (int i = 0; i < N; i ++ )
        for (int j = 0; j < N; j ++ ) {
            A[i][j] = random(1, 1e6);
            B[i][j] = random(1, 1e6);
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

static inline double *PTR(double *m, const int i, const int j) {
    return m + i * N + j;
}

void do_block(double * __restrict matA, double * __restrict matB, double * __restrict matC,
    const int block_size, const int unroll, const int stride) {
    __m512d c[unroll];
    for (int i = 0; i < block_size; i ++ ) {
        for (int j = 0; j < block_size; j += stride) {
            for (int u = 0; u < unroll; u ++ ) {
                c[u] = _mm512_loadu_pd(PTR(matC, i, j + u * 8));
            }
            for (int k = 0; k < block_size; k ++ ) {
                auto a = _mm512_set1_pd(*PTR(matA, i, k));
                for (int u = 0; u < unroll; u ++ ) {
                    auto b = _mm512_loadu_pd(PTR(matB, k, j + u * 8));
                    c[u] = _mm512_fmadd_pd(a, b, c[u]);
                }
            }
            for (int u = 0; u < unroll; u ++ ) {
                _mm512_storeu_pd(PTR(matC, i, j + u * 8), c[u]);
            }
        }
    }
}

// subword parallelism +
// instruction parallelism +
// tiling
void mat_subword_instruction_parallsim_tiled() {
    double *matA = reinterpret_cast<double *>(A);
    double *matB = reinterpret_cast<double *>(B);
    double *matC = reinterpret_cast<double *>(C);
    constexpr static int unroll = 4;
    constexpr static int block_size = 64;
    constexpr static int stride = unroll * 8;
    assert(block_size % stride == 0);
    // if N is not multiple of block_size, we just need to pad the matrix with zeros
    assert(N % block_size == 0);
    for (int i = 0; i < N; i += block_size) {
        for (int j = 0; j < N; j += block_size) {
            for (int k = 0; k < N; k += block_size) {
                do_block(PTR(matA, i, k), PTR(matB, k, j), PTR(matC, i, j), block_size, unroll, stride);
            }
        }
    }
}

void correct_test() {
    mat_vanilla();
    mat_subword_instruction_parallsim_tiled();
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
    correct_test();
    return 0;
}