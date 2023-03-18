#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define DIMX 64
#define DIMY 64
#define DIMK 3

texture<float, 2, cudaReadModeElementType> texImage;
__constant__ float constKernel[DIMK * DIMK];

__global__ void conv2D(float* output, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    __shared__ float shImage[DIMY + DIMK - 1][DIMX + DIMK - 1];

    // Load input image data into shared memory
    int shX = threadIdx.x + DIMK / 2;
    int shY = threadIdx.y + DIMK / 2;
    shImage[shY][shX] = tex2D(texImage, x, y);
    if (threadIdx.x < DIMK / 2) {
        shImage[shY][shX - DIMK / 2] = tex2D(texImage, x - DIMK / 2, y);
        shImage[shY][shX + blockDim.x] = tex2D(texImage, x + blockDim.x, y);
    }
    if (threadIdx.y < DIMK / 2) {
        shImage[shY - DIMK / 2][shX] = tex2D(texImage, x, y - DIMK / 2);
        shImage[shY + blockDim.y][shX] = tex2D(texImage, x, y + blockDim.y);
    }
    if (threadIdx.x < DIMK / 2 && threadIdx.y < DIMK / 2) {
        shImage[shY - DIMK / 2][shX - DIMK / 2] = tex2D(texImage, x - DIMK / 2, y - DIMK / 2);
        shImage[shY + blockDim.y][shX - DIMK / 2] = tex2D(texImage, x - DIMK / 2, y + blockDim.y);
        shImage[shY - DIMK / 2][shX + blockDim.x] = tex2D(texImage, x + blockDim.x, y - DIMK / 2);
        shImage[shY + blockDim.y][shX + blockDim.x] = tex2D(texImage, x + blockDim.x, y + blockDim.y);
    }

    __syncthreads();

    // Compute convolution using shared memory
    float sum = 0;
    for (int kY = 0; kY < DIMK; ++kY) {
        for (int kX = 0; kX < DIMK; ++kX) {
            if (x + kX >= DIMX || y + kY >= DIMY) continue;
            sum += shImage[threadIdx.y + kY][threadIdx.x + kX] * constKernel[kY * DIMK + kX];
        }
    }

    output[y * width + x] = sum;
}



int main() 
{

    // Was not in the first version
    // Initialize input image and kernel dimensions
    // const int DIMX = 512;
    // const int DIMY = 512;
    // const int DIMK = 3;

    // Allocate host memory
    float* hImage = (float*)malloc(sizeof(float) * DIMX * DIMY);
    float* hKernel = (float*)malloc(sizeof(float) * DIMK * DIMK);
    float* hOutput = (float*)malloc(sizeof(float) * DIMX * DIMY);

    
    
    srand(time(NULL)); // Was not in the first version

    // Generate random input image and kernel
    for (int i = 0; i < DIMX * DIMY; i++) {
        hImage[i] = (float)(rand() % 16);
    }
    for (int i = 0; i < DIMK * DIMK; i++) {
        hKernel[i] = (float)(rand() % 16);
    }

    // Allocate device memory
    float* dImage, *dOutput;
    cudaMalloc(&dImage, sizeof(float) * DIMX * DIMY);
    cudaMalloc(&dOutput, sizeof(float) * DIMX * DIMY);

    // Copy input image and kernel to device memory
    cudaMemcpy(dImage, hImage, sizeof(float) * DIMX * DIMY, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(constKernel, hKernel, sizeof(float) * DIMK * DIMK);

    // Set texture parameters
    texImage.addressMode[0] = cudaAddressModeWrap;
    texImage.addressMode[1] = cudaAddressModeWrap;
    texImage.filterMode = cudaFilterModeLinear;
    texImage.normalized = true;

    // Bind input image data to texture
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    cudaBindTexture2D(NULL, texImage, dImage, channelDesc, DIMX, DIMY, sizeof(float) * DIMX);

    // Compute output image dimensions and thread block dimensions
    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((DIMX + threadsPerBlock.x - 1) / threadsPerBlock.x, (DIMY + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Invoke CUDA kernel
    conv2D<<<numBlocks, threadsPerBlock>>>(dOutput, DIMX, DIMY);

    // Copy output data from device to host
    cudaMemcpy(hOutput, dOutput, sizeof(float) * DIMX * DIMY, cudaMemcpyDeviceToHost);

    // Free device and host memory
    cudaUnbindTexture(texImage);
    cudaFree(dImage);
    cudaFree(dOutput);
    free(hImage);
    free(hKernel);
    free(hOutput);

    return 0;
}









// About the main function
// This code initializes the input image and kernel dimensions, generates random input image and kernel values, allocates device memory, copies input image and kernel to device memory, binds input image data to texture, computes output image dimensions and thread block dimensions, invokes the CUDA kernel, copies output data from device to host, frees device and host memory, and returns 0.
