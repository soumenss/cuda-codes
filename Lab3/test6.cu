#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
// #include <time.h>

#define BLOCK_SIZE 64
#define KERNEL_SIZE 3


__global__ void convolution2D_kernel(float* output, const float* input, const float* kernel,
                                    int dimx, int dimy, int dimk) {



    // Declare shared memory
    __shared__ float shared_input[BLOCK_SIZE + KERNEL_SIZE - 1][BLOCK_SIZE + KERNEL_SIZE - 1];
    __shared__ float shared_kernel[KERNEL_SIZE][KERNEL_SIZE];



    // Calculate input and output indices
    int x_in = threadIdx.x + blockIdx.x * blockDim.x;
    int y_in = threadIdx.y + blockIdx.y * blockDim.y;
    int x_out = x_in - (KERNEL_SIZE / 2);
    int y_out = y_in - (KERNEL_SIZE / 2);


    // Load shared memory for input and kernel

    if (x_in < dimx && y_in < dimy) {
        shared_input[threadIdx.x + KERNEL_SIZE / 2][threadIdx.y + KERNEL_SIZE / 2] = input[x_in + y_in * dimx];
    }
    if (threadIdx.x < KERNEL_SIZE && threadIdx.y < KERNEL_SIZE) {
        shared_kernel[threadIdx.x][threadIdx.y] = kernel[threadIdx.x + threadIdx.y * dimk];
    }


    // Synchronize threads to ensure shared memory is loaded
    __syncthreads();

    // Calculate output value

    float output_value = 0.0;
    if (x_out >= 0 && x_out < dimx && y_out >= 0 && y_out < dimy) {
        for (int i = 0; i < KERNEL_SIZE; i++) {
            for (int j = 0; j < KERNEL_SIZE; j++) {
                output_value += shared_input[threadIdx.x + i][threadIdx.y + j] * shared_kernel[i][j];
            }
        }
        output[x_out + y_out * dimx] = output_value;
    }
}

int main(int argc, char** argv) {

    // Parse input arguments
    int DIMX = atoi(argv[1]);
    int DIMY = atoi(argv[2]);
    int DIMK = atoi(argv[3]);

    // Allocate host memory
    float* input = (float*) malloc(DIMX * DIMY * sizeof(float));
    float* kernel = (float*) malloc(DIMK * DIMK * sizeof(float));
    float* output = (float*) malloc(DIMX * DIMY * sizeof(float));

    // Initialize host memory with random values
    // srand(time(NULL));
    for (int i = 0; i < DIMX * DIMY; i++) {
        input[i] = (float) rand() / RAND_MAX * 15.0;
    }
    for (int i = 0; i < DIMK * DIMK; i++) {
        kernel[i] = (float) rand() / RAND_MAX * 15.0;
    }

    // Allocate device memory
    float* d_input, *d_kernel, *d_output;
    cudaMalloc((void**) &d_input, DIMX * DIMY * sizeof(float));
    cudaMalloc((void**) &d_kernel, DIMK * DIMK * sizeof(float));
    cudaMalloc((void**) &d_output, DIMX * DIMY * sizeof(float));

    // Copy host memory

    cudaMemcpy(d_input, input, DIMX * DIMY * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, DIMK * DIMK * sizeof(float), cudaMemcpyHostToDevice);

    // Initialize thread block and kernel grid dimensions
    dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_dim((DIMX + BLOCK_SIZE - 1) / BLOCK_SIZE, (DIMY + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Invoke CUDA kernel
    convolution2D_kernel<<<grid_dim, block_dim>>>(d_output, d_input, d_kernel, DIMX, DIMY, DIMK);
    

    // Copy results from device to host
    cudaMemcpy(output, d_output, DIMX * DIMY * sizeof(float), cudaMemcpyDeviceToHost);



    // Deallocate device memory
    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);

/*


    // Print the input and output arrays
    printf("Input:\n");
    for (int i = 0; i < DIMY; i++) {
        for (int j = 0; j < DIMX; j++) {
            printf("%6.1f", input[i * DIMX + j]);
        }
        printf("\n");
    }

    printf("\nKernel:\n");
    for (int i = 0; i < DIMK; i++) {
        for (int j = 0; j < DIMK; j++) {
            printf("%6.1f", kernel[i * DIMK + j]);
        }
        printf("\n");
    }

    printf("\nOutput:\n");
    for (int i = 0; i < DIMY; i++) {
        for (int j = 0; j < DIMX; j++) {
            printf("%6.1f", output[i * DIMX + j]);
        }
        printf("\n");
    }

*/

    // Free host memory
    free(input);
    free(kernel);
    free(output);

    return 0;


}