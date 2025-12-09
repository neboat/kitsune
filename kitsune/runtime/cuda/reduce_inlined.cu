#include <__clang_cuda_builtin_vars.h>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
// #include <cuda/atomic>

#ifndef GEN_BITCODE

__device__ __host__ void id_sum(void *v) {
  *reinterpret_cast<int64_t *>(v) = 0;
}
__device__ __host__ void merge_sum(void *l, void *r) {
  *reinterpret_cast<int64_t *>(l) += *reinterpret_cast<int64_t *>(r);
}
using view_t = int64_t;
#endif

// NOLINTBEGIN(*-reserved-identifier)

// NOLINTBEGIN(*-use-using)
typedef void (*__cilk_identity_fn)(void *);
typedef void (*__cilk_reduce_fn)(void *, void *);
// NOLINTEND(*-use-using)

constexpr size_t KITCUDA_WARP_SIZE = warpSize;
constexpr size_t KITCUDA_MAX_N_WARPS = 32;

__attribute__((always_inline)) static inline __device__ void
__kitcuda_warp_reduce(uint32_t *view /* local */, uint32_t *temp /* local */,
                      size_t size /* must be multiples of 4 */,
                      __cilk_reduce_fn reduce) noexcept {
  const size_t lane_id = threadIdx.x % KITCUDA_WARP_SIZE;
  // uint32_t mask = ~0;
#pragma unroll
  for (size_t delta = 1; delta < KITCUDA_WARP_SIZE; delta *= 2) {
  // for (size_t delta = KITCUDA_WARP_SIZE / 2; delta >= 1; delta /= 2) {
    for (size_t i = 0; i < size / sizeof(uint32_t); i++) {
      temp[i] = __shfl_down_sync(0xFFFFFFFF, view[i], delta);
      // temp[i] = __shfl_down_sync(mask, view[i], delta);
    }
    if (lane_id + delta < KITCUDA_WARP_SIZE) {
    // if (lane_id < delta) {
      reduce(view, temp);
    }

    // mask >>= delta;
    // if (lane_id >= delta)
    //   break;
    // reduce(view, temp);
  }
}

template<typename T>
__attribute__((always_inline)) static inline __device__ void
__kitcuda_system_reduce(T* ptr, T *view, __cilk_reduce_fn reduce) noexcept {
  T oldv, newv;
  // printf("adding %ld to result\n", *view);
  do {
    newv = *view;
    oldv = *ptr;
    reduce(&newv, &oldv);
  } while (!__atomic_compare_exchange(ptr, &oldv, &newv, false,
                                      __ATOMIC_SEQ_CST, __ATOMIC_RELAXED));
  // } while (atomicCAS(ptr, oldv, newv) != oldv);
  // } while (!cuda::std::atomic_compare_exchange_strong_explicit(
  //     reinterpret_cast<cuda::std::atomic<T> *>(ptr), &old, *view,
  //     cuda::std::memory_order_acq_rel, cuda::std::memory_order_relaxed));
}

// Device-side mutex implementation using atomic operations
struct CudaMutex {
  int lock_status; // 0 for unlocked, 1 for locked

  __device__ void lock() {
    // Spin-wait until the lock is acquired
    while (atomicCAS(&lock_status, 0, 1) != 0) {
    }
    // int expected = 0, desired = 1;
    // while (!__scoped_atomic_compare_exchange(
    //     &lock_status, &expected, &desired, false, __ATOMIC_SEQ_CST,
    //     __ATOMIC_SEQ_CST, __MEMORY_SCOPE_SYSTEM)) {
    // }
  }

  __device__ void unlock() { lock_status = 0; }
  // __device__ void unlock() { atomicExch(&lock_status, 0); }
  // __device__ void unlock() {
  //   // int val = 0;
  //   __scoped_atomic_exchange_n(&lock_status, 0, __ATOMIC_SEQ_CST,
  //                              __MEMORY_SCOPE_SYSTEM);
  // }
};

#ifdef GEN_BITCODE
extern "C"
#else
inline
#endif
    __attribute__((always_inline)) __device__ void __kitcuda_reduce(
        uint32_t *view /* local */, uint32_t *temp /* local */,
        uint32_t *shmem /* shared */, size_t size /* must be multiples of 4 */,
        uint32_t *result /* global result */,
        CudaMutex *mutex /* global mutex */, __cilk_identity_fn identity,
        __cilk_reduce_fn reduce) noexcept {
  const size_t lane_id = threadIdx.x % KITCUDA_WARP_SIZE;
  const size_t warp_id = threadIdx.x / KITCUDA_WARP_SIZE;
  const size_t n_warps = blockDim.x / KITCUDA_WARP_SIZE;
  // Reduce the views in the current warp.
  __kitcuda_warp_reduce(view, temp, size, reduce);
  // view[0] holds the inclusive sum of threads in the current warp.
  if (lane_id == 0) {
    for (size_t i = 0; i < size / sizeof(uint32_t); ++i)
      // When writing to shmem, use interleaved layout to eliminate bank
      // conflict on reads.
      shmem[i * KITCUDA_MAX_N_WARPS + warp_id] = view[i];
  }
  // TODO: Check if this is necessary in bitcode
  __syncthreads();

  // Reduce the views from all warps in the current block.
  if (warp_id == 0) {
    if (lane_id < n_warps) {
      for (size_t i = 0; i < size / sizeof(uint32_t); i++) {
        // No bank conflict here :)
        view[i] = shmem[i * KITCUDA_MAX_N_WARPS + lane_id];
      }
    } else {
      identity(view);
    }
    __kitcuda_warp_reduce(view, temp, size, reduce);

    __threadfence();

    if (threadIdx.x == 0) {
      if (size == 4) {
        __kitcuda_system_reduce(result, view, reduce);
      } else if (size == 8) {
        __kitcuda_system_reduce(reinterpret_cast<unsigned long long *>(result),
                                reinterpret_cast<unsigned long long *>(view),
                                reduce);
        // } else if (size == 16) {
        //   __kitcuda_system_reduce(reinterpret_cast<__int128 *>(result),
        //                          reinterpret_cast<__int128 *>(view), reduce);
      } else {
        // TODO: Support efficient non-constant time reduce functions,
        // associative reduce functions.

        mutex->lock();
        __threadfence();
        reduce(result, view);
        __threadfence();
        mutex->unlock();
      }
    }
  }
}
// NOLINTEND(*-reserved-identifier)

#ifndef GEN_BITCODE
__global__ void kernel_test(size_t n, view_t *result, CudaMutex *mutex) {
  __shared__ uint8_t shmem[sizeof(view_t) * KITCUDA_MAX_N_WARPS];

  view_t view, temp; // NOLINT(*-init*)
  // id_range(&view);                    // Compiler should generate this.
  id_sum(&view);

  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx == 0) {
    // printf("view %p, temp %p\n", (void*)&view, (void*)&temp);
    // result->start = 0;
    // result->end = 1;
    mutex->lock_status = 0;
  }

  if (idx < n) {
    // view.start = idx;
    // view.end = idx + 1;
    view = 1;
  }
  // __syncthreads();
  __kitcuda_reduce(
      reinterpret_cast<uint32_t *>(&view), reinterpret_cast<uint32_t *>(&temp),
      reinterpret_cast<uint32_t *>(shmem), sizeof(view_t),
      reinterpret_cast<uint32_t *>(result), mutex, id_sum, merge_sum);
  // if (idx == 0) {
  //   __threadfence();
  //   if (result->start != 0 || result->end != n) {
  //     printf("range at %ld: [%ld, %ld)\n", idx, result->start, result->end);
  //     assert(result->start == 0 && result->end == n);
  //   }
  // }
  // if (threadIdx.x == 0)
  //   printf("[%d] result %ld\n", blockIdx.x, *result);
}

#define CUDA_CHECK(X)                                                          \
  do {                                                                         \
    cudaError_t code = (X);                                                    \
    if (code != cudaSuccess) {                                                 \
      fprintf(stderr, "CUDA error (%s:%d): %s\n", __FILE__, __LINE__,          \
              cudaGetErrorString(code));                                       \
      exit(code);                                                              \
    }                                                                          \
  } while (0)

int main(int argc, char **argv) {
  if (argc != 2) {
    fprintf(stderr, "usage: kitcuda-scan <number of iterations>\n");
    exit(1);
  }
  constexpr size_t BLOCK_SIZE = 1024;

  size_t n = strtoul(argv[1], nullptr, 10);
  size_t n_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

  for (size_t i = 0; i<1000; i++) {
    view_t *result = nullptr;
    CUDA_CHECK(cudaMallocManaged(&result, sizeof(view_t)));
    CudaMutex *mutex = nullptr;
    CUDA_CHECK(cudaMalloc(&mutex, sizeof(CudaMutex)));
    kernel_test<<<n_blocks, BLOCK_SIZE>>>(n, result, mutex);
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("iter %ld: result %ld\n", i, *result);
    CUDA_CHECK(cudaFree(result));
    CUDA_CHECK(cudaFree(mutex));
  }
  return 0;
}
#endif