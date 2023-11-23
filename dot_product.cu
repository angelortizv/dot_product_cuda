/*******************************************************************************
  File: dot_product.cu

  Description:
    This program demonstrates a simple CUDA implementation for computing the dot
    product of two vectors using random values. It utilizes the cuRAND library
    for generating random numbers on the GPU.

  Author: Angelo Ortiz Vega - @angelortizv
  Date: November 22, 2023

  Notes:
    - Ensure that the appropriate CUDA toolkit is installed.
    - Compile with nvcc: nvcc dot_product.cu -o dot_product

  Usage:
    ./dot_product

*******************************************************************************/
#include <iostream>
#include <curand_kernel.h>

// Function to measure GPU time
void checkCudaErrors(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error at " << file << ":" << line << " code=" << static_cast<unsigned int>(result) << " \"" << func << "\"\n";
        exit(EXIT_FAILURE);
    }
}

#define checkCudaErrors(val) checkCudaErrors((val), #val, __FILE__, __LINE__)

// Function to measure GPU time
float measureKernelTime(cudaEvent_t start, cudaEvent_t stop) {
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    return elapsedTime;
}

__global__ void initializeRandom(curandState *state, unsigned long long seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, tid, 0, &state[tid]);
}

__global__ void dotProduct(curandState *state, float *a, float *b, int N, float *result) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < N; i += stride) {
        // Generate random numbers for vectors a and b
        a[i] = curand_uniform(&state[tid]);
        b[i] = curand_uniform(&state[tid]);
        result[tid] += a[i] * b[i]; // Calculate the dot product and accumulate the result
    }
}

int main() {
    const int N = 6000;

    float *a, *b, *result;
    a = new float[N];
    b = new float[N];
    result = new float[N];

    const int blockSize = 256;
    const int numBlocks = (N + blockSize - 1) / blockSize;

    // Allocate GPU memory
    float *d_a, *d_b, *d_result;
    curandState *d_state;
    cudaMalloc((void **)&d_a, N * sizeof(float));
    cudaMalloc((void **)&d_b, N * sizeof(float));
    cudaMalloc((void **)&d_result, N * sizeof(float));
    cudaMalloc((void **)&d_state, N * sizeof(curandState));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Initialize random number generator states on the GPU
    initializeRandom<<<numBlocks, blockSize>>>(d_state, time(NULL));

    // Launch GPU kernel
    dotProduct<<<numBlocks, blockSize>>>(d_state, d_a, d_b, N, d_result);

    // Measure GPU kernel execution time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapsedTime = measureKernelTime(start, stop);
    std::cout << "GPU Kernel Execution Time: " << elapsedTime << " ms " << "N=" << N << "\n";

    // Transfer the result from the GPU to the CPU
    cudaMemcpy(result, d_result, N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);
    cudaFree(d_state);

    delete[] a;
    delete[] b;
    delete[] result;

    return 0;
}
