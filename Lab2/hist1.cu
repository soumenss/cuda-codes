#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define THREAD_BLOCK_SIZE 64

__global__ void histogram(int* input, int* output, int VecDim, int BinNum)
{
    // Allocate shared memory for the s_hist
    __shared__ int s_hist[256];

    // Initialize shared memory s_hist to 0
    for (int i = threadIdx.x; i < BinNum; i += blockDim.x)
    {
        s_hist[i] = 0;
    }
    __syncthreads();

    // Compute the histogram
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < VecDim)
    {
        int bin = input[idx] % BinNum;
        atomicAdd(&s_hist[bin], 1);
    }
    __syncthreads();

    // Copy the histogram from shared memory to global memory
    for (int i = threadIdx.x; i < BinNum; i += blockDim.x)
    {
        atomicAdd(&output[i], s_hist[i]);
    }
}

int main(int argc, char** argv)
{
    // Parse input parameters
    int vecDim = atoi(argv[2]);
    int BinNum = atoi(argv[1]);

    // Allocate host memory
    int* h_input = (int*)malloc(vecDim * sizeof(int));
    int* h_output = (int*)malloc(BinNum * sizeof(int));

    // Generate random input vector
    for (int i = 0; i < vecDim; i++)
    {
        h_input[i] = rand() % 1024;
    }

    // Allocate device memory
    int* d_input, *d_output;
    cudaMalloc(&d_input, vecDim * sizeof(int));
    cudaMalloc(&d_output, BinNum * sizeof(int));

    // Copy host memory to device
    cudaMemcpy(d_input, h_input, vecDim * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0, BinNum * sizeof(int));

    // Initialize thread block and kernel grid dimensions
    dim3 blockDim(THREAD_BLOCK_SIZE, 1, 1);
    dim3 gridDim((vecDim + THREAD_BLOCK_SIZE - 1) / THREAD_BLOCK_SIZE, 1, 1);

    // Invoke CUDA kernel
    histogram<<<gridDim, blockDim>>>(d_input, d_output, vecDim, BinNum);

    // Copy results from device to host
    cudaMemcpy(h_output, d_output, BinNum * sizeof(int), cudaMemcpyDeviceToHost);

    // Print results
    for (int i = 0; i < BinNum; i++)
    {
        printf("%d ", h_output[i]);
    }
    printf("\n");

    // Deallocate device memory
    cudaFree(d_input);
    cudaFree(d_output);

    // Deallocate host memory
    free(h_input);
    free(h_output);

    return 0;
}
