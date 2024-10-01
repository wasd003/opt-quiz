#include <iostream>
#include <vector>
#include <queue>
#include <stack>
#include <deque>
#include <list>
#include <cassert>
#include <algorithm>
#include <utility>
#include <memory>
#include <chrono>
#include <random>

#define UNUSED(x) (void)(x)

static inline std::mt19937 generator(std::chrono::system_clock::now().time_since_epoch().count());

static inline int random(int minv, int maxv) {
    std::uniform_int_distribution<int> distribution(minv, maxv);
    return distribution(generator);
}

template<clockid_t ClockID>
struct ClockWatch {
    timespec beginSpec;

    ClockWatch() {
        clock_gettime(ClockID, &beginSpec);
    }

    double duration(timespec beforeTs, timespec afterTs) {
        return afterTs.tv_sec - beforeTs.tv_sec + 1e-9 * (afterTs.tv_nsec - beforeTs.tv_nsec);
    }

    double Get() {
        timespec cur;
        clock_gettime(ClockID, &cur);
        return duration(beginSpec, cur);
    }
};

