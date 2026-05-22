#include <stdio.h>
__global__ void hello() {
    printf("GPU thread %d says hello\n", threadIdx.x);
}
int main() {
    hello<<<1, 4>>>();
    cudaDeviceSynchronize();
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device: %s\n", prop.name);
    printf("SMs: %d  |  Mem: %.0f MB\n", prop.multiProcessorCount, prop.totalGlobalMem/1e6);
    return 0;
}
