#include <iostream>
#include <cstring>
#include <unordered_set>
#include <quiz/base.h>

constexpr static int TEST_SIZE = (1UL << 24);
constexpr static int TEST_STRUCT_SIZE = 48;

template<typename T, size_t Capacity, typename SecondHashFn>
requires requires(T t, SecondHashFn second_hashfn) {
    requires std::is_same_v<decltype(second_hashfn(t)), char>;
}
struct hash_set {
    std::vector<T> data;
    std::vector<char> control;
    std::hash<T> first_hasher;
    SecondHashFn second_hasher;
    size_t size {0};

    constexpr static int bucket_size = 256;

private:
    size_t first_hash(const T& val) const {
        const auto v = first_hasher(val) % Capacity;
        return v / bucket_size * bucket_size;
    }

    char second_hash(const T& val) const {
        return second_hasher(val);
    }

    struct find_result {
        size_t idx;
        size_t hash_val1;
        char hash_val2;
    };

    find_result find(const T& key) const {
        const auto hash_val1 = first_hash(key);
        const auto hash_val2 = second_hash(key);
        for (size_t i = hash_val1; ; i = (i + 1) % Capacity) {
            const auto c = control[i];
            if (!c)
                return {i, hash_val1, hash_val2};
            if ((c == hash_val2) && (data[i] == key))
                return {i, hash_val1, hash_val2};
        }
    }

public:
    hash_set() {
        data.resize(Capacity);
        control.resize(Capacity);
    }

    bool insert(const T& key) {
        assert(size < Capacity);
        auto [idx, hv1, hv2] = find(key);
        // const auto diff = (idx - hv1 + Capacity) % Capacity;
        // printf("diff:%ld hv1:%ld hv2:%d\n", diff, hv1, hv2);
        if (!control[idx]) { /// unexist key
            control[idx] = hv2;
            data[idx] = key;
            size ++;
            return true;
        } 
        return false;
    }

    bool remove(const T& key) {
        assert(size < Capacity);
        auto [idx, hv1, hv2] = find(key);
        if (control[idx]) {
            control[idx] = 0;
            size --;
            return true;
        }
        return false;
    }

    bool count(const T& key) const {
        auto [idx, hv1, hv2] = find(key);
        return (control[idx] != 0);
    }

    size_t get_size() const {
        return size;
    }
};

struct test_struct {
    char data[TEST_STRUCT_SIZE];
    bool operator==(const test_struct& rhs) const {
        for (int i = 0; i < TEST_STRUCT_SIZE; i ++ ) {
            if (data[i] != rhs.data[i])
                return false;
        }
        return true;
    }
};

template<>
struct std::hash<test_struct> {
    size_t operator()(const test_struct& t) const {
        size_t ret = 0;
        constexpr static int P = 133331;
        for (int i = 0; i < TEST_STRUCT_SIZE; i ++ ) {
            ret = ret * P + t.data[i];
        }
        return ret;
    }
};

struct second_hashfn {
    char operator()(const test_struct& rhs) const {
        return rhs.data[0];
    };
};

test_struct test_data[TEST_SIZE];

void init() {
    for (int i = 0; i < TEST_SIZE; i ++ ) {
        for (int j = 0; j < TEST_STRUCT_SIZE; j ++ ) {
            test_data[i].data[j] = random(1, 255);
        }
    }
}

void perf_test() {
    // hash_set<test_struct, TEST_SIZE << 1, second_hashfn> h;
    std::unordered_set<test_struct> h;
    // test insert
    for (int i = 0; i < TEST_SIZE; i ++ ) {
        h.insert(test_data[i]);
        auto r = h.count(test_data[i]);
        if (!r) {
            printf("ERROR: i:%d\n", i);
            return;
        }
    }
}

int main() {
    init();
    perf_test();
    return 0;
}