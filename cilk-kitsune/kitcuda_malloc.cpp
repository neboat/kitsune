#include <new>

// #define TRACK_MALLOC

#ifdef TRACK_MALLOC
#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <dlfcn.h>
#include <limits>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>
#include <x86intrin.h>
#endif

// NOLINTBEGIN(*-reserved-identifier, *-use-trailing-return-type, *-type-vararg,
// *-avoid-non-const-global-variables)

extern "C" {
void *__kitcuda_mem_alloc_managed(std::size_t size);

void __kitcuda_mem_free(void *ptr);

void *__kitcuda_mem_calloc_managed(std::size_t count, std::size_t elemsize);

void *__kitcuda_mem_realloc_managed(void *ptr, std::size_t size);

#ifdef TRACK_MALLOC
void *__libc_malloc(std::size_t size);

void __libc_free(void *ptr);

void *__libc_realloc(void *ptr, std::size_t size);
#endif
}

namespace {

#ifdef TRACK_MALLOC
template <typename T> class GlibcAllocator {
  public:
    // Standard allocator typedefs
    using value_type = T;
    using pointer = T *;
    using const_pointer = const T *;
    using reference = T &;
    using const_reference = const T &;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    // Rebind allocator to type U
    template <typename U> struct rebind {
        using other = GlibcAllocator<U>;
    };

    // Default constructor
    GlibcAllocator() noexcept {}

    // Copy constructor
    template <typename U> GlibcAllocator(const GlibcAllocator<U> &) noexcept {}

    // Allocate memory
    pointer allocate(size_type n) {
        if (n > max_size()) {
            throw std::bad_alloc();
        }

        if (n == 0) {
            return nullptr;
        }

        void *ptr = __libc_malloc(n * sizeof(T));
        if (!ptr) {
            throw std::bad_alloc();
        }

        return static_cast<pointer>(ptr);
    }

    // Deallocate memory
    void deallocate(pointer p, size_type) noexcept { __libc_free(p); }

    // Maximum number of objects that can be allocated
    size_type max_size() const noexcept {
        return std::numeric_limits<size_type>::max() / sizeof(T);
    }

    // Construct object at given address
    template <typename U, typename... Args>
    void construct(U *p, Args &&...args) {
        ::new (static_cast<void *>(p)) U(std::forward<Args>(args)...);
    }

    // Destroy object at given address
    template <typename U> void destroy(U *p) { p->~U(); }
};

// Comparison operators
template <typename T, typename U>
bool operator==(const GlibcAllocator<T> &, const GlibcAllocator<U> &) noexcept {
    return true;
}

template <typename T, typename U>
bool operator!=(const GlibcAllocator<T> &, const GlibcAllocator<U> &) noexcept {
    return false;
}

bool initialized = false;
std::mutex mutex;
struct AllocationData {
    uint64_t total_latency;
    uint32_t count;
};
using AllocationMap = std::unordered_map<
    std::size_t, AllocationData, std::hash<std::size_t>,
    std::equal_to<std::size_t>,
    GlibcAllocator<std::pair<const std::size_t, AllocationData>>>;

AllocationMap *allocations = nullptr;

void at_exit_handler() {
    if (!initialized) {
        return;
    }
    std::lock_guard<std::mutex> lock(mutex);
    std::vector<std::pair<std::size_t, AllocationData>,
                GlibcAllocator<std::pair<std::size_t, AllocationData>>>
        sorted_allocations;
    sorted_allocations.reserve(allocations->size());
    for (const auto &[size, data] : *allocations) {
        sorted_allocations.emplace_back(size, data);
    }
    std::sort(sorted_allocations.begin(), sorted_allocations.end(),
              [](const auto &a, const auto &b) { return a.first < b.first; });
    // Get the name of the shared object we're in
    Dl_info info;
    void *addr = (void *)&at_exit_handler;
    if (dladdr(addr, &info) != 0 && info.dli_fname != nullptr) {
        printf("Malloc stats for: %s\n", info.dli_fname);
    } else {
        printf("Malloc stats (unable to determine shared object)\n");
    }
    printf("      size,    count,      latency\n");
    for (const auto &[size, data] : sorted_allocations) {
        printf("%10zu, %8u, %12lu\n", size, data.count,
               data.total_latency / data.count);
    }

    allocations->~AllocationMap();
    __libc_free(allocations);
}

void try_initialize() {
    if (!initialized) {
        std::lock_guard<std::mutex> lock(mutex);
        if (!initialized) {
            atexit(at_exit_handler);
            allocations = static_cast<AllocationMap *>(
                __libc_malloc(sizeof(AllocationMap)));
            new (allocations) AllocationMap();
            initialized = true;
        }
    }
}

#endif

void *allocate(std::size_t size) {
#ifdef TRACK_MALLOC
    try_initialize();
    unsigned long long start = __rdtsc();
#endif

    const auto ret = __kitcuda_mem_alloc_managed(size);

#ifdef TRACK_MALLOC
    unsigned long long end = __rdtsc();
    const auto latency =
        end > start ? end - start
                    : (std::numeric_limits<unsigned long long>::max() - start) +
                          end + 1;
    {
        std::lock_guard<std::mutex> lock(mutex);
        if (auto it = allocations->find(size); it != allocations->end()) {
            it->second.total_latency += latency;
            it->second.count++;
        } else {
            allocations->insert({size, {latency, 1}});
        }
    }
#endif
    return ret;
}

void deallocate(void *ptr) { __kitcuda_mem_free(ptr); }
} // namespace

#if __cplusplus >= 201103L
#define NOEXCEPT_SINCE_CXX11 noexcept
#define NOEXCEPT_OR_THROW noexcept
#else
#define NOEXCEPT_SINCE_CXX11
#define NOEXCEPT_OR_THROW throw()
#endif

// =====================================================================================
// Operator new
// =====================================================================================

// Replaceable allocation functions

void *operator new(std::size_t size) { return allocate(size); }

void *operator new[](std::size_t size) { return allocate(size); }

#if __cplusplus >= 201703L
void *operator new(std::size_t size, std::align_val_t) {
    return allocate(size);
}

void *operator new[](std::size_t size, std::align_val_t) {
    return allocate(size);
}
#endif

// Replaceable non-throwing allocation functions

void *operator new(std::size_t size,
                   const std::nothrow_t &) NOEXCEPT_SINCE_CXX11 {
    return allocate(size);
}

void *operator new[](std::size_t size,
                     const std::nothrow_t &) NOEXCEPT_SINCE_CXX11 {
    return allocate(size);
}

#if __cplusplus >= 201703L
void *operator new(std::size_t size, std::align_val_t,
                   const std::nothrow_t &) noexcept {
    return allocate(size);
}

void *operator new[](std::size_t size, std::align_val_t,
                     const std::nothrow_t &) noexcept {
    return allocate(size);
}
#endif

// =====================================================================================
// Operator delete
// =====================================================================================

// Replaceable usual deallocation functions

void operator delete(void *ptr) NOEXCEPT_OR_THROW { __kitcuda_mem_free(ptr); }

void operator delete[](void *ptr) NOEXCEPT_OR_THROW { __kitcuda_mem_free(ptr); }

#if __cplusplus >= 201703L
void operator delete(void *ptr, std::align_val_t) noexcept { deallocate(ptr); }

void operator delete[](void *ptr, std::align_val_t) noexcept {
    deallocate(ptr);
}
#endif

#if __cplusplus >= 201402L
void operator delete(void *ptr, std::size_t) noexcept { deallocate(ptr); }

void operator delete[](void *ptr, std::size_t) noexcept { deallocate(ptr); }
#endif

#if __cplusplus >= 201703L
void operator delete(void *ptr, std::size_t, std::align_val_t) noexcept {
    deallocate(ptr);
}

void operator delete[](void *ptr, std::size_t, std::align_val_t) noexcept {
    deallocate(ptr);
}
#endif

// Replaceable placement deallocation functions

void operator delete(void *ptr, const std::nothrow_t &) NOEXCEPT_OR_THROW {
    deallocate(ptr);
}

void operator delete[](void *ptr, const std::nothrow_t &) NOEXCEPT_OR_THROW {
    deallocate(ptr);
}

#if __cplusplus >= 201703L
void operator delete(void *ptr, std::size_t, std::align_val_t,
                     const std::nothrow_t &) noexcept {
    deallocate(ptr);
}

void operator delete[](void *ptr, std::size_t, std::align_val_t,
                       const std::nothrow_t &) noexcept {
    deallocate(ptr);
}
#endif

// NOLINTEND(*-reserved-identifier, *-use-trailing-return-type, *-type-vararg,
// *-avoid-non-const-global-variables)
