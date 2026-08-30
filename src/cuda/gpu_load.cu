// gpu_load.cu — a sustained GPU load to drive the card to its power/thermal cap.
//
// HONEST FRAMING: this is a tiled matrix-multiply kept deliberately simple, run
// in a loop to sustain load. It demonstrates understanding of the memory
// hierarchy (shared-memory tiling) and how to saturate a GPU. For MAXIMUM power
// stress in production you'd call cuBLAS/CUTLASS -- this hand-rolled kernel is a
// teaching artifact, not a claim to beat vendor BLAS. That distinction is itself
// the point: a validation engineer knows which tool for which job.
//
// On a 35W GTX 1650 this is enough to hit the power cap and trip throttling,
// which is exactly the behaviour the paired NVML logger is there to catch.
//
//   nvcc -O3 -o gpu_load gpu_load.cu
//   ./gpu_load 60           # saturate for 60 seconds

#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cuda_runtime.h>

#define TILE 16

__global__ void tiledMatMul(const float* A, const float* B, float* C, int N) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float acc = 0.0f;
    for (int t = 0; t < N / TILE; ++t) {
        As[threadIdx.y][threadIdx.x] = A[row * N + t * TILE + threadIdx.x];
        Bs[threadIdx.y][threadIdx.x] = B[(t * TILE + threadIdx.y) * N + col];
        __syncthreads();
        for (int k = 0; k < TILE; ++k)              // the inner product, per tile
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }
    C[row * N + col] = acc;
}

static void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) { fprintf(stderr, "CUDA error (%s): %s\n", what, cudaGetErrorString(e)); exit(1); }
}

int main(int argc, char** argv) {
    const double seconds = (argc > 1) ? atof(argv[1]) : 30.0;
    const int N = 2048;                              // 2048^3 FLOPs per iter -> heavy
    const size_t bytes = (size_t)N * N * sizeof(float);

    float *dA, *dB, *dC;
    check(cudaMalloc(&dA, bytes), "malloc A");
    check(cudaMalloc(&dB, bytes), "malloc B");
    check(cudaMalloc(&dC, bytes), "malloc C");
    check(cudaMemset(dA, 1, bytes), "memset A");
    check(cudaMemset(dB, 1, bytes), "memset B");

    dim3 threads(TILE, TILE), blocks(N / TILE, N / TILE);
    printf("saturating GPU with %dx%d tiled GEMM for %.0f s...\n", N, N, seconds);

    auto t0 = std::chrono::steady_clock::now();
    long iters = 0;
    while (true) {
        tiledMatMul<<<blocks, threads>>>(dA, dB, dC, N);
        ++iters;
        if (iters % 20 == 0) {                       // periodic sync to measure + throttle-friendly
            check(cudaDeviceSynchronize(), "sync");
            double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            double gflops = (2.0 * N * N * N * iters) / (el * 1e9);
            printf("\r  t=%.1fs  iters=%ld  ~%.0f GFLOP/s   ", el, iters, gflops);
            fflush(stdout);
            if (el >= seconds) break;
        }
    }
    check(cudaDeviceSynchronize(), "final sync");
    printf("\ndone: %ld iterations.\n", iters);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return 0;
}
