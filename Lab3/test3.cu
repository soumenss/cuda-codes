#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define TILE_SIZE 32

texture<float, 2, cudaReadModeElementType> input_tex;
texture<float, 2, cudaReadModeElementType> kernel_tex;

__global__ void convolutionKernel(float* output, int dimX, int dimY, int dimK) {
    __shared__ float inputTile[TILE_SIZE][TILE_SIZE];

    int outputIdx = blockIdx.y * blockDim.y * dimX + blockIdx.x * blockDim.x + threadIdx.y * dimX + threadIdx.x;
    int inputIdx = blockIdx.y * blockDim.y * dimX + blockIdx.x * blockDim.x + threadIdx.y * dimX + threadIdx.x;

    float sum = 0.0f;

    for (int kRow = 0; kRow < dimK; kRow++) {
        for (int kCol = 0; kCol < dimK; kCol++) {
            int inputRow = blockIdx.y * blockDim.y + threadIdx.y - dimK / 2 + kRow;
            int inputCol = blockIdx.x * blockDim.x + threadIdx.x - dimK / 2 + kCol;

            if (inputRow >= 0 && inputRow < dimY && inputCol >= 0 && inputCol < dimX) {
                if (threadIdx.y < TILE_SIZE && threadIdx.x < TILE_SIZE) {
                    inputTile[threadIdx.y][threadIdx.x] = tex2D(input_tex, inputCol, inputRow);
                }
                __syncthreads();

                if (threadIdx.y < dimK && threadIdx.x < dimK) {
                    sum += inputTile[threadIdx.y][threadIdx.x] * tex2D(kernel_tex, kCol, kRow);
                }
                __syncthreads();
            }
        }
    }

    if (outputIdx < dimX * dimY) {
        output[outputIdx] = sum;
    }
}

void convolution(float* input, float* kernel, float* output, int dimX, int dimY, int dimK) {
    // Allocate device memory for input, kernel, and output arrays
    float* d_input, * d_kernel, * d_output;
    cudaMalloc(&d_input, dimX * dimY * sizeof(float));
    cudaMalloc(&d_kernel, dimK * dimK * sizeof(float));
    cudaMalloc(&d_output, dimX * dimY * sizeof(float));

    // Copy input and kernel data to device memory
    cudaMemcpy(d_input, input, dimX * dimY * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, dimK * dimK * sizeof(float), cudaMemcpyHostToDevice);

    // Set up texture memory
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    cudaBindTexture2D(NULL, input_tex, d_input, channelDesc, dimX, dimY, sizeof(float) * dimX);
    cudaBindTexture2D(NULL, kernel_tex, d_kernel, channelDesc, dimK, dimK, sizeof(float) * dimK);

    // Set up kernel launch configuration
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((dimX + TILE_SIZE - 1) / TILE_SIZE, (dimY + TILE_SIZE - 1) / TILE_SIZE);

    // Invoke CUDA kernel
    convolutionKernel<<<gridDim, blockDim>>>(d_output, dimX, , dimY, dimK);

    // Copy results from device to host
    cudaMemcpy(output, d_output, dimX * dimY * sizeof(float), cudaMemcpyDeviceToHost);

    // Clean up device memory
    cudaUnbindTexture(input_tex);
    cudaUnbindTexture(kernel_tex);
    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);

}



int main(int argc, char** argv) {
// Set up input and kernel sizes
    int dimX = atoi(argv[1]);
    int dimY = atoi(argv[2]);
    int dimK = atoi(argv[3]);


    // Allocate host memory for input, kernel, and output arrays
    float* input = (float*)malloc(dimX * dimY * sizeof(float));
    float* kernel = (float*)malloc(dimK * dimK * sizeof(float));
    float* output = (float*)malloc(dimX * dimY * sizeof(float));

    // Initialize input and kernel arrays with random values
    srand(time(NULL));
    for (int i = 0; i < dimX * dimY; i++) {
        input[i] = (float)(rand() % 16);
    }
    for (int i = 0; i < dimK * dimK; i++) {
        kernel[i] = (float)(rand() % 16);
    }

    // Perform convolution
    convolution(input, kernel, output, dimX, dimY, dimK);

    // Free host memory
    free(input);
    free(kernel);
    free(output);

    return 0;
}
