#include <ctime>
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <quiz/base.h>

using namespace std;

using LL = long long;

vector<LL> prefix_sum_baseline(const vector<int>& a) {
    const int n = a.size();
    vector<LL> pre(n);
    pre[0] = a[0];
    for (int i = 1; i < n; i ++ ) {
        pre[i] = pre[i - 1] + a[i];
    }
    return pre;
}

LL sum_baseline(const vector<int>& a) {
    LL sum = 0;
    for (auto x : a) sum += x;
    return sum;
}

vector<LL> prefix_sum_parallel(const vector<int>& a) {
    const int n = a.size();
    vector<LL> pre(n), cur(n);
    LL *p = pre.data();
    LL *c = cur.data();
    #pragma omp parallel for
    for (int i = 0; i < n; i ++ ) pre[i] = a[i];

    int step = 0, start;
    while ((start = (1 << step)) < n) {
#if 0
        #pragma omp parallel for
        for (int i = 0; i < n; i ++ ) {
            if (i >= start) {
                c[i] = p[i] + p[i - start];
            } else {
                c[i] = p[i];
            }
        }
#else
        // #pragma omp parallel for
        for (int i = start; i < n; i ++ ) {
            c[i] = p[i] + p[i - start];
        }
#endif
        step ++ ;
        swap(p, c);
    }
    return (step & 1 ? cur : pre);
}

LL sum_parallel(const vector<int>& nums) {
    // const int n = nums.size();
    const int n = 1e7;
    LL sum = 0;
    #pragma omp parallel for
    for (int i = 0; i < n; i ++ ) {
        LL local_sum = 0;
        local_sum += nums[i];
        #pragma omp atomic
        sum += local_sum;
    }
    return sum;
}

LL sum_reduction(const vector<int>& nums) {
    const int n = nums.size();
    LL sum = 0;
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < n; i ++ ) {
        sum += nums[i];
    }
    return sum;
}

pair<double, vector<LL>> test_baseline(const vector<int>& nums) {
    ClockWatch<CLOCK_MONOTONIC> clk;
    auto pre = prefix_sum_baseline(nums);
    return {clk.Get(), pre};
}

pair<double, vector<LL>> test_parallel(const vector<int>& nums) {
    ClockWatch<CLOCK_MONOTONIC> clk;
    auto pre = prefix_sum_parallel(nums);
    return {clk.Get(), pre};
}

void run_compare_test(const vector<int>& nums) {
    const auto &[a, pre_a] = test_baseline(nums);
    const auto &[b, pre_b] = test_parallel(nums);

    cout << "Baseline: " << a << " Parallel: " << b << endl;

    for (size_t i = 0; i < pre_a.size(); i ++ ) {
        if (pre_a[i] != pre_b[i]) {
            cerr << "Mismatch at " << i << ": " << pre_a[i] << " vs " << pre_b[i] << endl;
            return;
        }
    }
    cout << "Passed" << endl;
}

void run_single_test(const vector<int>& nums) {
    ClockWatch<CLOCK_REALTIME> clk;
    auto sum = sum_baseline(nums);
    cout << "Time: " << clk.Get() << " Sum:" << sum << endl;
}

int main() {
    int n;
    cin >> n;
    vector<int> nums(n);
    for (int i = 0; i < n; i ++ ) cin >> nums[i];

    run_single_test(nums);

    return 0;
}