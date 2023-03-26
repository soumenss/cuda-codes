#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Define the input, kernel, and output dimensions
#define DIM_X 512
#define DIM_Y 512
#define DIM_K 3

// Define the input and kernel arrays as textures
texture<float, 2, cudaReadModeElementType> input_tex;
texture<float, 2, cudaReadModeElementType> kernel_tex;

__global__ void convolutionKernel(float* input, float* kernel, float* output, int dimX, int dimY, int dimK) {
    // Declare shared memory for input and kernel
    __shared__ float input_shared[16][16];
    __shared__ float kernel_shared[3][3];

    // Calculate global index for current thread
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int globalIdy = blockIdx.y * blockDim.y + threadIdx.y;

    // Calculate the index for the shared memory
    int sharedIdx = threadIdx.x;
    int sharedIdy = threadIdx.y;

    // Copy the input image data to the shared memory
    input_shared[sharedIdy][sharedIdx] = tex2D(input_tex, globalIdx, globalIdy);

    // Copy the kernel data to the shared memory
    if (sharedIdy < dimK && sharedIdx < dimK) {
        kernel_shared[sharedIdy][sharedIdx] = kernel[sharedIdy * dimK + sharedIdx];
    }
    __syncthreads();

    float result = 0.0f;
    int radius = dimK / 2;

    // Iterate over the kernel mask
    for (int i = -radius; i <= radius; i++) {
        for (int j = -radius; j <= radius; j++) {
            // Calculate the index for the input and kernel
            int inputIdx = (globalIdy + i) * dimX + (globalIdx + j);
            int kernelIdx = (i + radius) * dimK + (j + radius);

            // Use texture memory for input and shared memory for kernel
            float inputVal = tex2D(input_tex, globalIdx + j, globalIdy + i);
            float kernelVal = kernel_shared[i + radius][j + radius];

            result += inputVal * kernelVal;
        }
    }

    // Write the output to global memory
    output[globalIdy * dimX + globalIdx] = result;
}

int main() {
    // Generate random input and kernel arrays
    float* input = (float*)malloc(sizeof(float) * DIM_X * DIM_Y);
    float* kernel = (float*)malloc(sizeof(float) * DIM_K * DIM_K);
    float* output = (float*)malloc(sizeof(float) * DIM_X * DIM_Y);

    for (int i = 0; i < DIM_X * DIM_Y; i++) {
        input[i] = (float)(rand() % 16);
    }

    for (int i = 0; i < DIM_K * DIM_K; i++) {
        kernel[i] = (float)(rand() % 16);
    }

    // Allocate device memory
    float* d_input, *d_kernel, *d_output;
    cudaMalloc(&d_input, sizeof(float) * DIM_X * DIM_Y);
    cudaMalloc(&d_kernel, sizeof(float) * DIM_K * DIM_K);
    cudaMalloc(&d_output, sizeof(float) * DIM_X * DIM_Y);

    // Copy input and kernel data to device memory
    cudaMemcpy(d_input, input, sizeof(float) * DIM_X * DIM_Y, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, sizeof(float) * DIM_K * DIM_K, cudaMemcpyHostToDevice);

    // Bind the textures to the input and kernel arrays
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    cudaBindTexture2D(NULL, input_tex, d_input, channelDesc, DIM_X, DIM_Y, sizeof(float) * DIM_X);
    cudaBindTexture2D(NULL, kernel_tex, d_kernel, channelDesc, DIM_K, DIM_K, sizeof(float) * DIM_K);

    // Define the thread block and kernel grid dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim((DIM_X + blockDim.x - 1) / blockDim.x, (DIM_Y + blockDim.y - 1) / blockDim.y);

    // Invoke the CUDA kernel
    convolutionKernel<<<gridDim, blockDim>>>(d_input, d_kernel, d_output, DIM_X, DIM_Y, DIM_K);

    // Copy the output data from device to host memory
    cudaMemcpy(output, d_output, sizeof(float) * DIM_X * DIM_Y, cudaMemcpyDeviceToHost);

    // Unbind the textures and deallocate device memory
    cudaUnbindTexture(input_tex);
    cudaUnbindTexture(kernel_tex);
    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);

    // Print the input and output arrays
    printf("Input:\n");
    for (int i = 0; i < DIM_Y; i++) {
        for (int j = 0; j < DIM_X; j++) {
            printf("%6.1f", input[i * DIM_X + j]);
        }
        printf("\n");
    }

    printf("\nKernel:\n");
    for (int i = 0; i < DIM_K; i++) {
        for (int j = 0; j < DIM_K; j++) {
            printf("%6.1f", kernel[i * DIM_K + j]);
        }
        printf("\n");
    }

    printf("\nOutput:\n");
    for (int i = 0; i < DIM_Y; i++) {
        for (int j = 0; j < DIM_X; j++) {
            printf("%6.1f", output[i * DIM_X + j]);
        }
        printf("\n");
    }

    // Deallocate host memory
    free(input);
    free(kernel);
    free(output);

    return 0;

}
