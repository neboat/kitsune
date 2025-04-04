
/*
 * Copyright (c) 2020 Triad National Security, LLC
 *                         All rights reserved.
 *
 * This file is part of the kitsune/llvm project.  It is released under
 * the LLVM license.
 */
#ifndef __KITSUNE_KITSUNE_H__
#define __KITSUNE_KITSUNE_H__

#include <stdint.h>
#include <stdio.h>
#include <stddef.h>

// #if defined(spawn)
// // FIXME KITSUNE: Should this be an error instead of a warning?
// #warning encountered multiple definitions of spawn!
// #else
// #define spawn _kitsune_spawn
// #endif

// #if defined(sync)
// // FIXME KITSUNE: Should this be an error instead of a warning?
// #warning encountered multiple definitions of sync!
// #else
// #define sync _kitsune_sync
// #endif

#if defined(forall)
// FIXME KITSUNE: Should this be an error instead of a warning?
#warning encountered multiple definitions of forall!
#else
#define forall _kitsune_forall
#endif

#if defined(_tapir_cuda_target)
  #ifdef __cplusplus
    extern "C" __attribute__((malloc))
    void* __kitcuda_mem_alloc_managed(size_t);
    template <typename T>
    inline __attribute__((always_inline))
      T* alloc(size_t N) {
      return (T*)__kitcuda_mem_alloc_managed(sizeof(T) * N);
    }
    inline __attribute__((always_inline))
    void *kit_malloc(size_t total_bytes) {
      return __kitcuda_mem_alloc_managed(total_bytes);
    }

    extern "C" void __kitcuda_mem_free(void*);
    template <typename T>
    void dealloc(T* array) {
      __kitcuda_mem_free((void*)array);
    }
    inline __attribute__((always_inline))
    void kit_free(void *array) {
      __kitcuda_mem_free(array);
    }
  #else
    void* __attribute__((malloc)) __kitcuda_mem_alloc_managed(size_t);
    inline __attribute__((always_inline))
    void *alloc(size_t total_bytes) {
      return __kitcuda_mem_alloc_managed(total_bytes);
    }

    void __kitcuda_mem_free(void*);
    inline __attribute__((always_inline))
    void dealloc(void *array) {
      __kitcuda_mem_free(array);
    }
  #endif
  extern "C" void* __attribute__((malloc)) __kitcuda_mem_realloc_managed(void *ptr, size_t size);
  inline __attribute__((always_inline))
  void *kit_realloc(void *ptr, size_t size) {
    // fprintf(stderr, "kit_realloc(%p, %ld) -> __kitcuda_mem_realloc_managed\n", ptr, size);
    return __kitcuda_mem_realloc_managed(ptr, size);
  }
#elif defined(_tapir_hip_target)
  #ifdef __cplusplus
    extern "C" __attribute__((malloc)) void* __kithip_mem_alloc_managed(size_t);
    template <typename T>
    inline __attribute__((always_inline))
      T* alloc(size_t N) {
      return (T*)__kithip_mem_alloc_managed(sizeof(T) * N);
    }

    extern "C" void __kithip_mem_free(void*);
    template <typename T>
    void dealloc(T* array) {
      __kithip_mem_free((void*)array);
    }
  #else
    void* __attribute__((malloc)) __kithip_mem_alloc_managed(size_t);
    inline __attribute__((always_inline))
    void *alloc(size_t total_bytes) {
      return __kithip_mem_alloc_managed(total_bytes);
    }

    void __kithip_mem_free(void*);
    inline __attribute__((always_inline))
    void dealloc(void *array) {
       __kithip_mem_free(array);
    }
  #endif
  extern "C" void* __attribute__((malloc)) __kithip_mem_realloc_managed(void *ptr, size_t size);
  inline __attribute__((always_inline))
  void *kit_realloc(void *ptr, size_t size) {
    return __kithip_mem_realloc_managed(ptr, size);
  }
#else
  #ifdef __cplusplus
    extern "C" __attribute__((malloc))
    void* __kitrt_default_mem_alloc(size_t);
    template <typename T>
    inline __attribute__((always_inline))
    T* alloc(size_t N) {
      return (T*)__kitrt_default_mem_alloc(sizeof(T) * N);
    }
    inline __attribute__((always_inline))
    void *kit_malloc(size_t total_bytes) {
      return __kitrt_default_mem_alloc(total_bytes);
    }

    extern "C" void __kitrt_default_mem_free(void*);
    template <typename T>
    void dealloc(T* array) {
      __kitrt_default_mem_free(array);
    }
    inline __attribute__((always_inline))
    void kit_free(void *array) {
      __kitrt_default_mem_free(array);
    }
  #else
    void* __attribute__((malloc)) __kitrt_default_mem_alloc(size_t);
    inline __attribute__((always_inline))
    void *alloc(size_t total_bytes) {
      return __kitrt_default_mem_alloc(total_bytes);
    }

    void __kitrt_default_mem_free(void*);
    inline __attribute__((always_inline))
    void dealloc(void* array) {
       __kitrt_default_mem_free(array);
    }
  #endif // __cplusplus
  extern "C" void* __attribute__((malloc)) __kitrt_default_mem_realloc(void *, size_t);
  inline __attribute__((always_inline))
  void *kit_realloc(void *ptr, size_t size) {
    // fprintf(stderr, "kit_realloc(%p, %ld) -> __kitrt_default_mem_realloc\n", ptr, size);
    return __kitrt_default_mem_realloc(ptr, size);
  }
#endif // cpu targets

#endif // __KITSUNE_KITSUNE_H__
