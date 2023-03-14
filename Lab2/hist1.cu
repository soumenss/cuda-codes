#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 256

__global__ void histogram(int* input, int* output, int inputSize, int binNum)
{
    // Allocate shared memory for the bins
    __shared__ int bins[256];

    // Initialize shared memory bins to 0
    for (int i = threadIdx.x; i < binNum; i += blockDim.x)
    {
        bins[i] = 0;
    }
    __syncthreads();

    // Compute the histogram
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < inputSize)
    {
        int bin = input[tid] % binNum;
        atomicAdd(&bins[bin], 1);
    }
    __syncthreads();

    // Copy the histogram from shared memory to global memory
    for (int i = threadIdx.x; i < binNum; i += blockDim.x)
    {
        atomicAdd(&output[i], bins[i]);
    }
}

int main(int argc, char** argv)
{
    // Parse input parameters
    int vecDim = atoi(argv[1]);
    int binNum = atoi(argv[2]);

    // Allocate host memory
    int* h_input = (int*)malloc(vecDim * sizeof(int));
    int* h_output = (int*)malloc(binNum * sizeof(int));

    // Generate random input vector
    for (int i = 0; i < vecDim; i++)
    {
        h_input[i] = rand() % 1024;
    }

    // Allocate device memory
    int* d_input, *d_output;
    cudaMalloc(&d_input, vecDim * sizeof(int));
    cudaMalloc(&d_output, binNum * sizeof(int));

    // Copy host memory to device
    cudaMemcpy(d_input, h_input, vecDim * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0, binNum * sizeof(int));

    // Initialize thread block and kernel grid dimensions
    dim3 blockDim(THREADS_PER_BLOCK, 1, 1);
    dim3 gridDim((vecDim + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, 1, 1);

    // Invoke CUDA kernel
    histogram<<<gridDim, blockDim>>>(d_input, d_output, vecDim, binNum);

    // Copy results from device to host
    cudaMemcpy(h_output, d_output, binNum * sizeof(int), cudaMemcpyDeviceToHost);

    // Print results
    for (int i = 0; i < binNum; i++)
    {
        printf("bin %d: %d\n", i, h_output[i]);
    }

    // Deallocate device memory
    cudaFree(d_input);
    cudaFree(d_output);

    // Deallocate host memory
    free(h_input);
    free(h_output);

    return 0;
}
