#include <iostream>
#include <vector>
#include <atomic>
#include <cstring>
#include <quiz/base.h>
#include <quiz/static_queue.h>

using namespace std;

using PII = pair<int, int>;

pair<vector<int>, vector<int>> build_graph() {
    int n, m;
    cin >> n >> m;
    vector<int> out_degree(n + 5);
    vector<PII> all_edges;
    vector<int> offsets(n + 5);
    vector<int> edges(2 * m + 5);
    vector<int> idxs(n + 5);
    all_edges.reserve(m);
    for (int i = 0, a, b; i < m; i ++ ) {
        cin >> a >> b;
        all_edges.emplace_back(a, b);
        out_degree[a] ++ ;
        out_degree[b] ++ ;
    }
    for (int i = 1; i <= n + 1; i ++ ) {
        idxs[i] = offsets[i] = offsets[i - 1] + out_degree[i - 1];
    }
    for (auto& [a, b] : all_edges) {
        edges[idxs[a] ++ ] = b;
        edges[idxs[b] ++ ] = a;
    }
    return {offsets, edges};
}

int bfs(const vector<int>& offsets, const vector<int>& edges) {
    #define N (100010)

    const int n = offsets.size();
    int frontier[N];
    int frontier_count = 1;
    vector<atomic<bool>> st(n);
    frontier[0] = 1;
    st[1] = true;
    int cnt = 1;
    
    while (frontier_count) {
        bool first_thread = true;
        #pragma omp parallel for
        for (int i = 0; i < frontier_count; i ++ ) {
            int local_count = 0;
            int local_frontier[N];
            bool flag;
            const auto cur = frontier[i];
            for (int i = offsets[cur]; i < offsets[cur + 1]; i ++ ) {
                auto ne = edges[i];
                if (!(flag = st[ne]) && st[ne].compare_exchange_strong(flag, true)) {
                    local_frontier[local_count ++ ] = ne;
                }
            }
            #pragma omp critical
            {
                if (first_thread) {
                    first_thread = false;
                    frontier_count = 0;
                }
                memcpy(reinterpret_cast<void *>(frontier + frontier_count), reinterpret_cast<void*>(local_frontier), local_count * sizeof(int));
                frontier_count += local_count;
                cnt += local_count;
            }
        }
    }
    return cnt;
}


int main() {
    const auto [offsets, edges] = build_graph();

    ClockWatch<CLOCK_MONOTONIC> clk;
    const auto cnt = bfs(offsets, edges);
    const auto t = clk.Get();
    std::cout << "count:" << cnt << " time:" << t << endl;
    return 0;
}