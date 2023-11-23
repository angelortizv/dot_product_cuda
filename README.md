# CUDA Dot Product

## Overview

This CUDA program, implemented in `dot_product.cu`, showcases a parallelized computation of the dot product of two vectors on the GPU using random values. The cuRAND library is employed to generate random numbers on the GPU.

## Author

- **Angelo Ortiz Vega** - [@angelortizv](https://github.com/angelortizv)

## Date

- November 22, 2023

## Prerequisites

- Ensure that the appropriate CUDA toolkit is installed.
- Compile with nvcc: `nvcc dot_product.cu -o dot_product`

## Usage

Execute the compiled program:

```bash
./dot_product
```

## Implementation Details

- The program dynamically allocates memory for vectors `a`, `b`, and `result`.
- GPU memory is allocated for `d_a`, `d_b`, `d_result`, and `d_state`.
- The kernel functions `initializeRandom` and `dotProduct` are responsible for initializing random number generator states and computing the dot product, respectively.
- GPU kernel execution time is measured using CUDA events.
- The final result is transferred from the GPU to the CPU.
- Memory allocated on the GPU is freed, and dynamic memory on the CPU is deallocated.

## Performance

The program demonstrates the parallelization benefits of GPU computation for the dot product, and it reports the GPU kernel execution time in milliseconds.

## Repository

Find the complete project on [GitHub](https://github.com/angelortizv/dot_product_cuda).

## Notes

- For CUDA error checking and measuring GPU time, utility functions `checkCudaErrors` and `measureKernelTime` are provided.
- The random seed for initializing the cuRAND generator is based on the current time.
- The block size and the number of blocks are configured for optimal GPU utilization.

Feel free to explore and modify the code to suit your specific requirements.
