#include <quiz/base.h>

using namespace std;

constexpr static int N = 2048;

double A[N][N];
double B[N][N];
double C[N][N];
double ans[N][N];

void init() {
    for (int i = 0; i < N; i ++ )
        for (int j = 0; j < N; j ++ ) {
            A[i][j] = random(1, 1e6);
            B[i][j] = random(1, 1e6);
            C[i][j] = 0;
        }
}

void mat_vanilla() {
    for (int i = 0; i < N; i ++ ) {
        for (int j = 0; j < N; j ++ ) {
            for (int k = 0; k < N; k ++ ) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

void mat_interchange_loop() {
    for (int i = 0; i < N; i ++ ) {
        for (int k = 0; k < N; k ++ ) {
            for (int j = 0; j < N; j ++ ) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

void mat_parallel_loop() {
    #pragma omp parallel for
    for (int i = 0; i < N; i ++ ) {
        for (int k = 0; k < N; k ++ ) {
            for (int j = 0; j < N; j ++ ) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

void mat_tiling() {
    const int s = 64;
    #pragma omp parallel for
    for (int i0 = 0; i0 < N; i0 += s)
        #pragma omp parallel for
        for (int j0 = 0; j0 < N; j0 += s)
            for (int k0 = 0; k0 < N; k0 += s)
                for (int i1 = 0; i1 < s; i1 ++ )
                    for (int k1 = 0; k1 < s; k1 ++ )
                        for (int j1 = 0; j1 < s; j1 ++ )
                            C[i0 + i1][j0 + j1] = A[i0 + i1][k0 + k1] * B[k0 + k1][j0 + j1];
}

void mat_tiling_2layer() {
    const int s = 256;
    const int t = 32;
    #pragma omp parallel for
    for (int i0 = 0; i0 < N; i0 += s)
        #pragma omp parallel for
        for (int j0 = 0; j0 < N; j0 += s)
            for (int k0 = 0; k0 < N; k0 += s)
                for (int i1 = 0; i1 < s; i1 += t)
                    for (int j1 = 0; j1 < s; j1 += t)
                        for (int k1 = 0; k1 < s; k1 += t)
                            for (int i2 = 0; i2 < t; i2 ++ )
                                for (int k2 = 0; k2 < t; k2 ++ )
                                    for (int j2 = 0; j2 < t; j2 ++ )
                            C[i0 + i1+ i2][j0 + j1 + j2] = A[i0 + i1 + i2][k0 + k1 + k2] * B[k0 + k1 + k2][j0 + j1 + j2];
}

// TODO: use openmp to accelerate
void dfs(double *a, double *b, double *c, const int n, const int base) {
    #define ele(ptr, r, c) (ptr + r * N + c)

    if (n <= base) {
        for (int i = 0; i < n; i ++ )
            for (int k = 0; k < n; k ++ )
                for (int j = 0; j < n; j ++ )
                    *ele(c, i, j) += *ele(a, i, k) * *ele(b, k, j);
        return;
    }

    #pragma omp parallel sections
    {
        #pragma omp section
        dfs(ele(a, 0, 0), ele(b, 0, 0), ele(c, 0, 0), n / 2, base);

        #pragma omp section
        dfs(ele(a, 0, 0), ele(b, 0, n / 2), ele(c, 0, n / 2), n / 2, base);

        #pragma omp section
        dfs(ele(a, n / 2, 0), ele(b, 0, 0), ele(c, n / 2, 0), n / 2, base);

        #pragma omp section
        dfs(ele(a, n / 2, 0), ele(b, 0, n / 2), ele(c, n / 2, n / 2), n / 2, base);
    }

    #pragma omp parallel sections
    {
        #pragma omp section
        dfs(ele(a, 0, n / 2), ele(b, n / 2, 0), ele(c, 0, 0), n / 2, base);

        #pragma omp section
        dfs(ele(a, 0, n / 2), ele(b, n / 2, n / 2), ele(c, 0, n / 2), n / 2, base);

        #pragma omp section
        dfs(ele(a, n / 2, n / 2), ele(b, n / 2, 0), ele(c, n / 2, 0), n / 2, base);

        #pragma omp section
        dfs(ele(a, n / 2, n / 2), ele(b, n / 2, n / 2), ele(c, n / 2, n / 2), n / 2, base);
    }
}

void mat_tiling_multi_layer() {
    dfs((double *)A, (double *)B, (double *)C, N, 32);
}



int main() {
    init();

    ClockWatch<CLOCK_REALTIME> clock;
    mat_tiling_multi_layer();
    auto tv = clock.Get();
    std::cout << tv << std::endl;
    return 0;
}
