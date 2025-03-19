#ifndef CLANGD
#include <cilk/cilk.h>
#else
#define cilk_for for
#define cilk_reducer(id, merge)
#endif

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {
// void matmul(const std::vector<float> &A, const std::vector<float> &B,
//             std::vector<float> &C, size_t N) {
//     cilk_for (size_t i = 0; i < N; i++) {
//         [[tapir::target("cuda")]] cilk_for (size_t j = 0; j < N; j++) {
//             C[i * N + j] = 0;
//             for (size_t k = 0; k < N; k++) {
//                 C[i * N + j] += A[i * N + k] * B[k * N + j];
//             }
//         }
//     }
// }

void id_sum(void *data) { *reinterpret_cast<float *>(data) = 0; }

void merge_sum(void *data, void *other) {
    *reinterpret_cast<float *>(data) += *reinterpret_cast<float *>(other);
}

void matmul(const std::vector<float> &A, const std::vector<float> &B,
            std::vector<float> &C, size_t N) {
    cilk_for (size_t i = 0; i < N; i++) {
        cilk_for (size_t j = 0; j < N; j++) {
            float cilk_reducer(id_sum, merge_sum) sum = 0;
            [[tapir::target("cuda")]] cilk_for (size_t k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

void matmul_serial(const std::vector<float> &A, const std::vector<float> &B,
                   std::vector<float> &C, size_t N) {
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < N; j++) {
            C[i * N + j] = 0;
            for (size_t k = 0; k < N; k++) {
                C[i * N + j] += A[i * N + k] * B[k * N + j];
            }
        }
    }
}
} // namespace

// extern void *__kitcuda_mem_alloc_managed(size_t size);
// extern void __kitcuda_mem_free(void *ptr);

int main(int argc, char **argv) {
    if (argc != 2) {
        printf("Usage: %s <N>\n", argv[0]);
        return 1;
    }

    size_t N = strtoul(argv[1], NULL, 10);

    std::vector<float> A(N * N);
    std::vector<float> B(N * N);
    std::vector<float> C(N * N);
    std::vector<float> C_serial(N * N);
    // Randomly initialize A and B
    srand(42);
    for (size_t i = 0; i < N * N; i++) {
        A[i] = (float)rand() / (float)RAND_MAX;
        B[i] = (float)rand() / (float)RAND_MAX;
    }

    printf("Starting matmul\n");
    matmul(A, B, C, N);
    printf("Starting matmul_serial\n");
    matmul_serial(A, B, C_serial, N);

    float abs_diff = 0;
    for (size_t i = 0; i < N * N; i++) {
        abs_diff = fmaxf(abs_diff, fabsf(C[i] - C_serial[i]));
    }
    printf("Absolute difference: %f\n", abs_diff);
    return 0;
}
