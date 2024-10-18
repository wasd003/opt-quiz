#include <ctime>
#include <iostream>
#include <vector>
#include <quiz/base.h>

using namespace std;

using PII = pair<int, int>;

decltype(auto) build_graph() {
    int n, m;
    cin >> n >> m;
    vector<vector<int>> g(n + 5);
    for (int i = 0, a, b; i < m; i ++ ) {
        cin >> a >> b;
        g[a].push_back(b);
        g[b].push_back(a);
    }
    return g;
}

int bfs(const vector<vector<int>>& g) {
    queue<int> q;
    const int n = g.size();
    vector<bool> st(n);
    q.push(1);
    st[1] = true;
    int cnt = 0;
    while (q.size()) {
        auto cur = q.front(); q.pop();
        cnt ++ ;
        for (auto ne : g[cur]) {
            if (!st[ne]) {
                st[ne] = true;
                q.push(ne);
            }
        }
    }
    return cnt;
}

int main() {
    const auto g = build_graph();

    ClockWatch<CLOCK_MONOTONIC> clk;
    const auto cnt = bfs(g);
    const auto t = clk.Get();
    std::cout << "count:" << cnt << " time:" << t << endl;
    return 0;
}