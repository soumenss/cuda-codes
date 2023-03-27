#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
// #include <time.h>

// #define BLOCK_SIZE 16
// #define KERNEL_SIZE 3
#define TILE_WIDTH 16
#define MAX_KERNEL_RADIUS 3

//__global__ void convolution2D_kernel(float* output, const float* input, const float* kernel,
 //                                   int dimx, int dimy, int dimk) {
__global__ void convolution2D_kernel(float *in, float *out, float *kernel, int dimX, int dimY, int dimK) {
    // Determine row and column indices of current thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Define the radius of the kernel
    int radius = dimK / 2;

    // Allocate shared memory for the tile of input image
    __shared__ float tile[TILE_WIDTH + MAX_KERNEL_RADIUS * 2][TILE_WIDTH + MAX_KERNEL_RADIUS * 2];

    // Load tile from global memory to shared memory
    int tileRow = threadIdx.y + radius;
    int tileCol = threadIdx.x + radius;
    if (row < dimY && col < dimX) {
        tile[tileRow][tileCol] = in[row * dimX + col];
    }
    if (threadIdx.y < radius && row < dimY && col < dimX) {
        // Load padding rows from global memory to shared memory
        tile[tileRow - radius][tileCol] = in[(row - radius) * dimX + col];
        tile[tileRow + blockDim.y][tileCol] = in[(row + blockDim.y) * dimX + col];
    }
    if (threadIdx.x < radius && row < dimY && col < dimX) {
        // Load padding columns from global memory to shared memory
        tile[tileRow][tileCol - radius] = in[row * dimX + col - radius];
        tile[tileRow][tileCol + blockDim.x] = in[row * dimX + col + blockDim.x];
    }
    if (threadIdx.x < radius && threadIdx.y < radius) {
        // Load padding corners from global memory to shared memory
        tile[tileRow - radius][tileCol - radius] = in[(row - radius) * dimX + col - radius];
        tile[tileRow - radius][tileCol + blockDim.x] = in[(row - radius) * dimX + col + blockDim.x];
        tile[tileRow + blockDim.y][tileCol - radius] = in[(row + blockDim.y) * dimX + col - radius];
        tile[tileRow + blockDim.y][tileCol + blockDim.x] = in[(row + blockDim.y) * dimX + col + blockDim.x];
    }

    // Synchronize to make sure the tile is fully loaded
    __syncthreads();

    // Compute the convolution result for the current thread
    float sum = 0;
    for (int i = 0; i < dimK; i++) {
        for (int j = 0; j < dimK; j++) {
            int rowIdx = tileRow + i - radius;
            int colIdx = tileCol + j - radius;
            sum += kernel[i * dimK + j] * tile[rowIdx][colIdx];
        }
    }

    // Write the result to output array
    if (row < dimY && col < dimX) {
        out[row * dimX + col] = sum;
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
    convolution2D_kernel<<<grid_dim, block_dim>>>(d_input, d_output, d_kernel, DIMX, DIMY, DIMK);
    

    // Copy results from device to host
    cudaMemcpy(output, d_output, DIMX * DIMY * sizeof(float), cudaMemcpyDeviceToHost);



    // Deallocate device memory
    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);




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



    // Free host memory
    free(input);
    free(kernel);
    free(output);

    return 0;


}