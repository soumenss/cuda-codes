#include <cuda_runtime.h>
// #include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
// #include <time.h>

#define BLOCK_SIZE 16
#define KERNEL_SIZE 3

__global__ void convolution2D_kernel(int* output, int output_width, int output_height,
                                      const int* input, int input_width, int input_height,
                                      const int* kernel, int kernel_size) {

    // Declare shared memory
    __shared__ int shared_input[BLOCK_SIZE + KERNEL_SIZE - 1][BLOCK_SIZE + KERNEL_SIZE - 1];
    __shared__ int shared_kernel[KERNEL_SIZE][KERNEL_SIZE];

    // Calculate input and output indices
    int x_in = threadIdx.x + blockIdx.x * blockDim.x;
    int y_in = threadIdx.y + blockIdx.y * blockDim.y;
    int x_out = x_in - (KERNEL_SIZE / 2);
    int y_out = y_in - (KERNEL_SIZE / 2);

    // Load shared memory for input and kernel
    if (x_in < input_width && y_in < input_height) {
        shared_input[threadIdx.x + KERNEL_SIZE / 2][threadIdx.y + KERNEL_SIZE / 2] = input[x_in + y_in * input_width];
    }
    if (threadIdx.x < KERNEL_SIZE && threadIdx.y < KERNEL_SIZE) {
        shared_kernel[threadIdx.x][threadIdx.y] = kernel[threadIdx.x + threadIdx.y * kernel_size];
    }

    // Synchronize threads to ensure shared memory is loaded
    __syncthreads();

    // Calculate output value
    int output_value = 0.0;
    if (x_out >= 0 && x_out < output_width && y_out >= 0 && y_out < output_height) {
        for (int i = 0; i < KERNEL_SIZE; i++) {
            for (int j = 0; j < KERNEL_SIZE; j++) {
                output_value += shared_input[threadIdx.x + i][threadIdx.y + j] * shared_kernel[i][j];
            }
        }
        output[x_out + y_out * output_width] = output_value;
    }
}

int main(int argc, char** argv) {

    // Parse input arguments
    int input_width = atoi(argv[1]);
    int input_height = atoi(argv[2]);
    int kernel_size = atoi(argv[3]);

    // Allocate host memory
    int* input = (int*) malloc(input_width * input_height * sizeof(int));
    int* kernel = (int*) malloc(kernel_size * kernel_size * sizeof(int));
    int* output = (int*) malloc(input_width * input_height * sizeof(int));

    // Initialize host memory with random values
    // srand(time(NULL));
    for (int i = 0; i < input_width * input_height; i++) {
        // input[i] = (int) rand() / RAND_MAX * 5;
        input[i] = (int) rand() % (5+1);
    }
    for (int i = 0; i < kernel_size * kernel_size; i++) {
        kernel[i] = (int) rand() % (5+1);
    }

    // Allocate device memory
    int* d_input, *d_kernel, *d_output;
    cudaMalloc((void**) &d_input, input_width * input_height * sizeof(int));
    cudaMalloc((void**) &d_kernel, kernel_size * kernel_size * sizeof(int));
    cudaMalloc((void**) &d_output, input_width * input_height * sizeof(int));

    // Copy host memory

    cudaMemcpy(d_input, input, input_width * input_height * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, kernel_size * kernel_size * sizeof(int), cudaMemcpyHostToDevice);

    // Initialize thread block and kernel grid dimensions
    dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_dim((input_width + BLOCK_SIZE - 1) / BLOCK_SIZE, (input_height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Invoke CUDA kernel
    convolution2D_kernel<<<grid_dim, block_dim>>>(d_output, input_width, input_height, d_input, input_width, input_height, d_kernel, kernel_size);

    // Copy results from device to host
    cudaMemcpy(output, d_output, input_width * input_height * sizeof(int), cudaMemcpyDeviceToHost);

    // Deallocate device memory
    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);

    // Print the input and output arrays
    printf("Input:\n");
    for (int i = 0; i < input_height; i++) {
        for (int j = 0; j < input_width; j++) {
            printf("%d", input[i * input_width + j]);
        }
        printf("\n");
    }

    printf("\nKernel:\n");
    for (int i = 0; i < kernel_size; i++) {
        for (int j = 0; j < kernel_size; j++) {
            printf("%d", kernel[i * kernel_size + j]);
        }
        printf("\n");
    }

    printf("\nOutput:\n");
    for (int i = 0; i < input_height; i++) {
        for (int j = 0; j < input_width; j++) {
            printf("%d", output[i * input_width + j]);
        }
        printf("\n");
    }



    // Free host memory
    free(input);
    free(kernel);
    free(output);

    return 0;


}