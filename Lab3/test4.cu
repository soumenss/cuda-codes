#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16

__global__ void convolution2D_kernel(float* inputImage, float* kernel, float* outputImage, int dimX, int dimY, int dimK) {
    __shared__ float sharedKernel[16 * 16];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * blockDim.x + tx;
    int y = blockIdx.y * blockDim.y + ty;

    float sum = 0.0f;

    for (int kx = 0; kx < dimK; kx++) {
        for (int ky = 0; ky < dimK; ky++) {
            int i = x + kx - dimK / 2;
            int j = y + ky - dimK / 2;

            if (i >= 0 && i < dimX && j >= 0 && j < dimY) {
                sharedKernel[ty * dimK + tx] = kernel[ky * dimK + kx];
                __syncthreads();

                sum += inputImage[j * dimX + i] * sharedKernel[ty * dimK + tx];
                __syncthreads();
            }
        }
    }

    if (x < dimX && y < dimY) {
        outputImage[y * dimX + x] = sum;
    }
}


void convolution2D(float* inputImage, float* kernel, float* outputImage, int dimX, int dimY, int dimK) {
    // allocate device memory
    float* d_inputImage, *d_kernel, *d_outputImage;
    cudaMalloc((void**)&d_inputImage, dimX * dimY * sizeof(float));
    cudaMalloc((void**)&d_kernel, dimK * dimK * sizeof(float));
    cudaMalloc((void**)&d_outputImage, dimX * dimY * sizeof(float));

    // copy memory from host to device
    cudaMemcpy(d_inputImage, inputImage, dimX * dimY * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, dimK * dimK * sizeof(float), cudaMemcpyHostToDevice);

    // initialize thread block and kernel grid dimensions
    dim3 threadBlockSize(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 kernelGridSize((dimX + BLOCK_SIZE - 1) / BLOCK_SIZE, (dimY + BLOCK_SIZE - 1) / BLOCK_SIZE, 1);

    // invoke CUDA kernel
    convolution2D_kernel<<<kernelGridSize, threadBlockSize>>>(d_inputImage, d_kernel, d_outputImage, dimX, dimY, dimK);

    // copy memory from device to host
    cudaMemcpy(outputImage, d_outputImage, dimX * dimY * sizeof(float), cudaMemcpyDeviceToHost);

    // deallocate device memory
    cudaFree(d_inputImage);
    cudaFree(d_kernel);
    cudaFree(d_outputImage);
}

int main(int argc, char** argv) {
    // check command line arguments
    if (argc != 4) {
        printf("Usage: ./convolution2D <dimX> <dimY> <dimK>\n");
        return 1;
    }

    // parse command line arguments
    int dimX = atoi(argv[1]);
    int dimY = atoi(argv[2]);
    int dimK = atoi(argv[3]);

    // allocate memory for input 2D image and 2D kernel
    float* inputImage = (float*)malloc(dimX * dimY * sizeof(float));
    float* kernel = (float*)malloc(dimK * dimK * sizeof(float));

    // generate random values for input 2D image and 2D kernel
    for (int i = 0; i < dimX * dimY; i++) {
        inputImage[i] = (float)(rand() % 16);
    }

    for (int i = 0; i < dimK * dimK; i++) {
        kernel[i] = (float)(rand() % 16);
    }

    // allocate memory for output 2D image
    float* outputImage = (float*)malloc(dimX * dimY * sizeof(float));

    // perform 2D convolution
    convolution2D(inputImage, kernel, outputImage, dimX, dimY, dimK);



    // Print the input and output arrays
    printf("Input:\n");
    for (int i = 0; i < dimY; i++) {
        for (int j = 0; j < dimX; j++) {
            printf("%6.1f", inputImage[i * dimX + j]);
        }
        printf("\n");
    }

    printf("\nKernel:\n");
    for (int i = 0; i < dimK; i++) {
        for (int j = 0; j < dimK; j++) {
            printf("%6.1f", kernel[i * dimK + j]);
        }
        printf("\n");
    }

    printf("\nOutput:\n");
    for (int i = 0; i < dimY; i++) {
        for (int j = 0; j < dimX; j++) {
            printf("%6.1f", outputImage[i * dimX + j]);
        }
        printf("\n");
    }


    // deallocate memory
    free(inputImage);
    free(kernel);
    free(outputImage);

    return 0;
}