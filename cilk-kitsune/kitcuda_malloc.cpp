#include <cstdlib>
#include <new>

// NOLINTBEGIN(*-reserved-identifier, *-use-trailing-return-type)

extern "C" {
void *__kitcuda_mem_alloc_managed(size_t size);

void __kitcuda_mem_free(void *ptr);

void *__kitcuda_mem_calloc_managed(size_t count, size_t elemsize);

void *__kitcuda_mem_realloc_managed(void *ptr, size_t size);

void *__wrap_malloc(size_t size) { return __kitcuda_mem_alloc_managed(size); }

void __wrap_free(void *ptr) { __kitcuda_mem_free(ptr); }

void *__wrap_calloc(size_t count, size_t elemsize) {
    return __kitcuda_mem_calloc_managed(count, elemsize);
}

void *__wrap_realloc(void *ptr, size_t size) {
    return __kitcuda_mem_realloc_managed(ptr, size);
}
}

void *operator new(size_t size) { return __kitcuda_mem_alloc_managed(size); }

void *operator new[](size_t size) { return __kitcuda_mem_alloc_managed(size); }

void operator delete(void *ptr) noexcept { __kitcuda_mem_free(ptr); }

void operator delete(void *ptr, size_t) noexcept { __kitcuda_mem_free(ptr); }

void operator delete[](void *ptr) noexcept { __kitcuda_mem_free(ptr); }

void operator delete[](void *ptr, size_t) noexcept { __kitcuda_mem_free(ptr); }

#if __cplusplus >= 201703L
void *operator new(size_t size, std::align_val_t) {
    return __kitcuda_mem_alloc_managed(size);
}

void *operator new[](size_t size, std::align_val_t) {
    return __kitcuda_mem_alloc_managed(size);
}

void operator delete(void *ptr, std::align_val_t) noexcept {
    __kitcuda_mem_free(ptr);
}

void operator delete[](void *ptr, std::align_val_t) noexcept {
    __kitcuda_mem_free(ptr);
}

void operator delete(void *ptr, size_t, std::align_val_t) noexcept {
    __kitcuda_mem_free(ptr);
}

void operator delete[](void *ptr, size_t, std::align_val_t) noexcept {
    __kitcuda_mem_free(ptr);
}
#endif

#undef HIDDEN

// NOLINTEND(*-reserved-identifier, *-use-trailing-return-type)