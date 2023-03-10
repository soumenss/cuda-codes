#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void histogram_kernel(int* d_in, int* d_hist, int bin_num, int vec_dim)
{
    extern __shared__ int s_hist[];

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = threadIdx.x; i < bin_num; i += blockDim.x) {
        s_hist[i] = 0;
    }

    __syncthreads();

    while (idx < vec_dim)
    {
        atomicAdd(&s_hist[d_in[idx] % bin_num], 1);
        idx += stride;
    }

    __syncthreads();

    for (int i = threadIdx.x; i < bin_num; i += blockDim.x) {
        atomicAdd(&d_hist[i], s_hist[i]);
    }
}

int main(int argc, char** argv)
{

    int vec_dim = atoi(argv[2]);
    int bin_num = atoi(argv[1]);


    // allocate host memory for inputs
    int* h_in = (int*)malloc(vec_dim * sizeof(int));
    int* h_hist = (int*)malloc(bin_num * sizeof(int));

    for (int i = 0; i < vec_dim; i++) {
        h_in[i] = rand() % 1024;
    }

    memset(h_hist, 0, bin_num * sizeof(int));

    int* d_in;
    int* d_hist;

    // allocate device memory
    cudaMalloc((void**)&d_in, vec_dim * sizeof(int));
    cudaMalloc((void**)&d_hist, bin_num * sizeof(int));

    // copy nputs from host memory to device
    cudaMemcpy(d_in, h_in, vec_dim * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_hist, h_hist, bin_num * sizeof(int), cudaMemcpyHostToDevice);

    int block_dim = 256;
    int grid_dim = (vec_dim + block_dim - 1) / block_dim;

    // invoke CUDA kernel
    histogram_kernel<<<grid_dim, block_dim, bin_num * sizeof(int)>>>(d_in, d_hist, bin_num, vec_dim);

    // copy results from device to host
    cudaMemcpy(h_hist, d_hist, bin_num * sizeof(int), cudaMemcpyDeviceToHost);

    // print results
    for (int i = 0; i < bin_num; i++) {
        printf("%d ", h_hist[i]);
    }
    printf("\n");

    // deallocate device memory
    cudaFree(d_in);
    cudaFree(d_hist);

    // deallocate host memory
    free(h_in);
    free(h_hist);

    return 0;
}
