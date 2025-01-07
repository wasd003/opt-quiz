#pragma once

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
#define force_inline __attribute__((__always_inline__))

static inline std::mt19937 generator(std::chrono::system_clock::now().time_since_epoch().count());

static inline int random(int minv, int maxv) {
    std::uniform_int_distribution<int> distribution(minv, maxv);
    return distribution(generator);
}

template<clockid_t ClockID>
struct ClockWatch {
    timespec beginSpec;

private:
    double duration(timespec beforeTs, timespec afterTs) {
        return afterTs.tv_sec - beforeTs.tv_sec + 1e-9 * (afterTs.tv_nsec - beforeTs.tv_nsec);
    }

public:
    ClockWatch() {
        clock_gettime(ClockID, &beginSpec);
    }

    double Get() {
        timespec cur;
        clock_gettime(ClockID, &cur);
        return duration(beginSpec, cur);
    }
};

/**
 * Returns aligned pointers when allocations are requested. Default alignment
 * is 64B = 512b, sufficient for AVX-512 and most cache line sizes.
 *
 * @tparam ALIGNMENT_IN_BYTES Must be a positive power of 2.
 */
template<typename    ElementType,
         std::size_t ALIGNMENT_IN_BYTES = 64>
class AlignedAllocator
{
private:
    static_assert(
        ALIGNMENT_IN_BYTES >= alignof( ElementType ),
        "Beware that types like int have minimum alignment requirements "
        "or access will result in crashes."
    );

public:
    using value_type = ElementType;
    static std::align_val_t constexpr ALIGNMENT{ ALIGNMENT_IN_BYTES };

    /**
     * This is only necessary because AlignedAllocator has a second template
     * argument for the alignment that will make the default
     * std::allocator_traits implementation fail during compilation.
     * @see https://stackoverflow.com/a/48062758/2191065
     */
    template<class OtherElementType>
    struct rebind
    {
        using other = AlignedAllocator<OtherElementType, ALIGNMENT_IN_BYTES>;
    };

public:
    constexpr AlignedAllocator() noexcept = default;

    constexpr AlignedAllocator( const AlignedAllocator& ) noexcept = default;

    template<typename U>
    constexpr AlignedAllocator( AlignedAllocator<U, ALIGNMENT_IN_BYTES> const& ) noexcept
    {}

    [[nodiscard]] ElementType*
    allocate( std::size_t nElementsToAllocate )
    {
        if ( nElementsToAllocate
             > std::numeric_limits<std::size_t>::max() / sizeof( ElementType ) ) {
            throw std::bad_array_new_length();
        }

        auto const nBytesToAllocate = nElementsToAllocate * sizeof( ElementType );
        return reinterpret_cast<ElementType*>(
            ::operator new[]( nBytesToAllocate, ALIGNMENT ) );
    }

    void
    deallocate(                  ElementType* allocatedPointer,
                [[maybe_unused]] std::size_t  nBytesAllocated )
    {
        /* According to the C++20 draft n4868 § 17.6.3.3, the delete operator
         * must be called with the same alignment argument as the new expression.
         * The size argument can be omitted but if present must also be equal to
         * the one used in new. */
        ::operator delete[]( allocatedPointer, ALIGNMENT );
    }
};

static inline void print_vec(auto&& data, auto&& name) {
    std::cout << name << ": ";
    std::for_each(data.begin(), data.end(), [](auto x) { std::cout << x << " "; });
    std::cout << std::endl;
}
#define PRINT_VEC(data) print_vec(data, #data)

static inline bool check_vec_eq(const std::vector<float>& v0, const std::vector<float>& v1) {
    if (v0.size() != v1.size()) {
        std::cout << "size mismatch" << v0.size() << "-" << v1.size() << " \n";
    }
    const auto [v0it, v1it] = std::mismatch(v0.begin(), v0.end(), v1.begin(), [](float a, float b) {
        return std::abs(a - b) < 1e-5;
    });
    if (v0it == v0.end()) {
        std::cout << "PASS" << std::endl;
    } else {
        std::cout << "FAIL" << std::endl;
        std::cout << "Index: " << std::distance(v0.begin(), v0it) << std::endl;
        std::cout << "v0: " << *v0it << " v1: " << *v1it << std::endl;
    }
    return true;

}