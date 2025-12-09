#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>

#ifndef GEN_BITCODE
struct range {
  int64_t start;
  int64_t end;
};

__device__ __host__ void id_range(void *v) {
  *reinterpret_cast<range *>(v) = {-1, -1};
}

__device__ __host__ void reduce_range(void *left_, void *right_) {
  auto *left = reinterpret_cast<range *>(left_);
  auto *right = reinterpret_cast<range *>(right_);
  // left->end = left->end == right->end ? right->end : left->start = -1;
  if (right->start == -1)
    return;
  if (left->start == -1) {
    *left = *right;
    return;
  }
  if (left->end != right->start) {
    printf("attempting to merge [%ld, %ld) with [%ld, %ld)\n", left->start,
           left->end, right->start, right->end);
    assert(left->end == right->start);
  }
  left->end = right->end;
}
#endif

// NOLINTBEGIN(*-reserved-identifier)

// NOLINTBEGIN(*-use-using)
typedef void (*__cilk_identity_fn)(void *);
typedef void (*__cilk_reduce_fn)(void *, void *);
// NOLINTEND(*-use-using)

constexpr size_t KITCUDA_WARP_SIZE = 32;
constexpr size_t KITCUDA_MAX_N_WARPS = 32;

constexpr size_t KITCUDA_SCAN_STATE_X = 0; // Invalid
constexpr size_t KITCUDA_SCAN_STATE_A = 1; // Aggregate available
constexpr size_t KITCUDA_SCAN_STATE_P = 2; // Prefix available

__attribute__((always_inline)) static inline __device__ void
__kitcuda_warp_scan(int32_t *view /* local */, int32_t *temp /* local */,
                    size_t size /* must be multiples of 4 */,
                    __cilk_identity_fn identity,
                    __cilk_reduce_fn reduce) noexcept {
  const size_t lane_id = threadIdx.x % KITCUDA_WARP_SIZE;
#pragma unroll
  for (size_t delta = 1; delta < KITCUDA_WARP_SIZE; delta *= 2) {
    for (size_t i = 0; i < size / sizeof(int32_t); i++)
      temp[i] = __shfl_up_sync(0xFFFFFFFF, view[i], delta);
    if (lane_id >= delta) {
      for (size_t i = 0; i < size / sizeof(int32_t); i++) {
        int32_t t = view[i];
        view[i] = temp[i];
        temp[i] = t;
      }
      reduce(view, temp);
    }
  }
}

#ifdef GEN_BITCODE
extern "C"
#else
inline
#endif
    __attribute__((always_inline)) __device__ void
    __kitcuda_scan(int32_t *view /* local */, int32_t *temp_1 /* local */,
                   int32_t *temp_2 /* local */, int32_t *temp_3 /* local */,
                   int32_t *shmem /* shared */, int32_t *aggregate /* global */,
                   int32_t *inclusive_prefix /* global */,
                   int32_t *scan_state /* global */,
                   size_t size /* must be multiples of 4 */,
                   int32_t *result /* global result */,
                   size_t n /* loop trip count */, __cilk_identity_fn identity,
                   __cilk_reduce_fn reduce) noexcept {
  const size_t lane_id = threadIdx.x % KITCUDA_WARP_SIZE;
  const size_t warp_id = threadIdx.x / KITCUDA_WARP_SIZE;
  const size_t n_warps = blockDim.x / KITCUDA_WARP_SIZE;
  __kitcuda_warp_scan(view, temp_1, size, identity, reduce);
  // view[lane_id] holds the inclusive sum of threads up to lane_id within the
  // current warp.
  if (lane_id == KITCUDA_WARP_SIZE - 1) {
    // So view here is the warp-wide sum. Write to shmem.
    for (size_t i = 0; i < size / sizeof(int32_t); i++) {
      // When writing to shmem, use interleaved layout to eliminate bank
      // conflict on reads.
      shmem[i * KITCUDA_MAX_N_WARPS + warp_id] = view[i];
    }
  }
  __syncthreads();
  if (warp_id == 0) {
    // Read from shmem
    if (lane_id < n_warps) {
      for (size_t i = 0; i < size / sizeof(int32_t); i++) {
        // No bank conflict here :)
        temp_3[i] = shmem[i * KITCUDA_MAX_N_WARPS + lane_id];
      }
    } else
      identity(temp_3);
    __kitcuda_warp_scan(temp_3, temp_1, size, identity, reduce);
    // temp_3[lane_id] holds the include sum of warps up to lane_id in the
    // current block.
    if (lane_id == KITCUDA_WARP_SIZE - 1) {
      // So temp_3 here is the block-wide sum. Write to global.
      int32_t *dst = aggregate + blockIdx.x * size / sizeof(int32_t);
      for (size_t i = 0; i < size / sizeof(int32_t); i++)
        dst[i] = temp_3[i];
      __threadfence();
      scan_state[blockIdx.x] = KITCUDA_SCAN_STATE_A;
      // Now kick off decoupled lookback
      identity(temp_2); // temp_2 is going to hold the running prefix sum
      identity(temp_1); // temp_1 is going to hold the value of temp_2 in
                        // the prior iteration
      for (size_t lookback_index = blockIdx.x; lookback_index--;) {
        int32_t state; // NOLINT(*-init-variables)
        while ((state = scan_state[lookback_index]) == KITCUDA_SCAN_STATE_X)
          __threadfence(); // Wait for the state to change
        int32_t *src =
            (state == KITCUDA_SCAN_STATE_A ? aggregate : inclusive_prefix) +
            lookback_index * size / sizeof(int32_t);
        __threadfence();
        for (size_t i = 0; i < size / sizeof(int32_t); i++)
          temp_2[i] = src[i];
        reduce(temp_2, temp_1);
        for (size_t i = 0; i < size / sizeof(int32_t); i++)
          temp_1[i] = temp_2[i];
        if (state == KITCUDA_SCAN_STATE_P)
          break; // Done!
      }
      reduce(temp_2, temp_3); // temp_2 modified
      dst = inclusive_prefix + blockIdx.x * size / sizeof(int32_t);
      for (size_t i = 0; i < size / sizeof(int32_t); i++)
        dst[i] = temp_2[i];
      __threadfence();
      scan_state[blockIdx.x] = KITCUDA_SCAN_STATE_P;
      // temp_1 still holds the exclusive prefix sum
    }
    // Broadcast the exclusive prefix sum to other warps
    for (size_t i = 0; i < size / sizeof(int32_t); i++)
      temp_1[i] = __shfl_sync(0xFFFFFFFF, temp_1[i], KITCUDA_WARP_SIZE - 1);
    __syncwarp();
    if (lane_id < n_warps) {
      size_t idx = 0;
      if (lane_id < n_warps - 1) {
        idx = lane_id + 1;
        reduce(temp_1, temp_3);
      }
      for (size_t i = 0; i < size / sizeof(int32_t); i++)
        shmem[i * KITCUDA_MAX_N_WARPS + idx] = temp_1[i];
    }
  }
  __syncthreads();
  for (size_t i = 0; i < size / sizeof(int32_t); i++)
    temp_1[i] = shmem[i * KITCUDA_MAX_N_WARPS + warp_id];
  reduce(temp_1, view);
  for (size_t i = 0; i < size / sizeof(int32_t); i++)
    view[i] = temp_1[i];
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (result && idx < n) {
    int32_t *dst = result + idx * size / sizeof(int32_t);
    for (size_t i = 0; i < size / sizeof(int32_t); i++)
      dst[i] = view[i];
  }
}
// NOLINTEND(*-reserved-identifier)

#ifndef GEN_BITCODE
__global__ void kernel_test(size_t n, range *aggregate, range *inclusive_prefix,
                            int32_t *scan_state) {
  __shared__ int8_t shmem[sizeof(range) * KITCUDA_MAX_N_WARPS];

  range view, temp_1, temp_2, temp_3; // NOLINT(*-init*)
  id_range(&view);                    // Compiler should generate this.

  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    view.start = idx;
    view.end = idx + 1;
  }

  __kitcuda_scan(
      reinterpret_cast<int32_t *>(&view), reinterpret_cast<int32_t *>(&temp_1),
      reinterpret_cast<int32_t *>(&temp_2),
      reinterpret_cast<int32_t *>(&temp_3), reinterpret_cast<int32_t *>(shmem),
      reinterpret_cast<int32_t *>(aggregate),
      reinterpret_cast<int32_t *>(inclusive_prefix), scan_state, sizeof(range),
      nullptr, n, id_range, reduce_range);
  if (idx < n) {
    if (view.start != 0 || view.end != idx + 1) {
      printf("range at %lu: [%ld, %ld)\n", idx, view.start, view.end);
      assert(false);
    }
  }
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

  for (size_t i = 0;; i++) {
    range *aggregate = nullptr;
    CUDA_CHECK(cudaMalloc(&aggregate, n_blocks * sizeof(range)));
    range *inclusive_prefix = nullptr;
    CUDA_CHECK(cudaMalloc(&inclusive_prefix, n_blocks * sizeof(range)));
    int32_t *scan_state = nullptr;
    CUDA_CHECK(cudaMalloc(&scan_state, n_blocks * sizeof(int32_t)));
    kernel_test<<<n_blocks, BLOCK_SIZE>>>(n, aggregate, inclusive_prefix,
                                          scan_state);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(aggregate));
    CUDA_CHECK(cudaFree(inclusive_prefix));
    CUDA_CHECK(cudaFree(scan_state));
    printf("iteration %lu done\n", i);
  }
  return 0;
}
#endif