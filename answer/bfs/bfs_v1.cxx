#include <iostream>
#include <vector>
#include <quiz/base.h>

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
    queue<int> q;
    const int n = offsets.size();
    vector<bool> st(n);
    q.push(1);
    st[1] = true;
    int cnt = 0;
    while (q.size()) {
        const auto cur = q.front(); q.pop();
        cnt ++ ;
        for (int i = offsets[cur]; i < offsets[cur + 1]; i ++ ) {
            auto ne = edges[i];
            if (!st[ne]) {
                st[ne] = true;
                q.push(ne);
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