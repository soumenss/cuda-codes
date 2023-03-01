#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void histogram(int *input, int *hist, int num_elements, int num_bins) {
    extern __shared__ int s_hist[];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int tsize = blockDim.x * gridDim.x;
    int i;
    for (i = threadIdx.x; i < num_bins; i += blockDim.x) {
        s_hist[i] = 0;
    }
    __syncthreads();
    while (tid < num_elements) {
        atomicAdd(&s_hist[input[tid]], 1);
        tid += tsize;
    }
    __syncthreads();
    for (i = threadIdx.x; i < num_bins; i += blockDim.x) {
        atomicAdd(&hist[i], s_hist[i]);
    }
}

int main(int argc, char **argv) {
    int vec_dim = atoi(argv[1]);
    int bin_num = atoi(argv[2]);
    int *input, *hist;
    int *d_input, *d_hist;
    int i;
    size_t input_size = vec_dim * sizeof(int);
    size_t hist_size = bin_num * sizeof(int);
    input = (int *)malloc(input_size);
    hist = (int *)malloc(hist_size);
    for (i = 0; i < vec_dim; i++) {
        input[i] = rand() % bin_num;
    }
    memset(hist, 0, hist_size);
    cudaMalloc((void **)&d_input, input_size);
    cudaMalloc((void **)&d_hist, hist_size);
    cudaMemcpy(d_input, input, input_size, cudaMemcpyHostToDevice);
    int block_size = 256;
    int grid_size = (vec_dim + block_size - 1) / block_size;
    int shared_mem_size = bin_num * sizeof(int);
    histogram<<<grid_size, block_size, shared_mem_size>>>(d_input, d_hist, vec_dim, bin_num);
    cudaMemcpy(hist, d_hist, hist_size, cudaMemcpyDeviceToHost);
    for (i = 0; i < bin_num; i++) {
        printf("%d ", hist[i]);
    }
    printf("\n");
    cudaFree(d_input);
    cudaFree(d_hist);
    free(input);
    free(hist);
    return 0;
}
