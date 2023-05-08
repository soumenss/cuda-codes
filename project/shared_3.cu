#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <stdio.h>

#include <cstdlib>
#include <ctime>

#include <iostream>
#include <fstream>

// Define image dimensions
// #define WIDTH 512
// #define HEIGHT 512

// Define TV regularization hyperparameters
// #define LAMBDA 0.01
// #define EPSILON 0.0001
// #define MAX_ITERATIONS 100

// Define block size
#define BLOCK_SIZE 64

// Define kernel function to calculate the gradient of an image
__global__ void gradient(float* image, float* grad_x, float* grad_y, int WIDTH, int HEIGHT)
{
    __shared__ float shared_image[BLOCK_SIZE][BLOCK_SIZE];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= 1 && x < WIDTH - 1 && y >= 1 && y < HEIGHT - 1)
    {
        shared_image[threadIdx.y][threadIdx.x] = image[y * WIDTH + x];
        __syncthreads();

        float dx = shared_image[threadIdx.y][threadIdx.x + 1] - shared_image[threadIdx.y][threadIdx.x - 1];
        float dy = shared_image[threadIdx.y + 1][threadIdx.x] - shared_image[threadIdx.y - 1][threadIdx.x];

        grad_x[y * WIDTH + x] = dx;
        grad_y[y * WIDTH + x] = dy;
    }
}

// Define kernel function to update the denoised image using the TV regularization technique
__global__ void update(float* image, float* grad_x, float* grad_y, int WIDTH, int HEIGHT, float LAMBDA, float EPSILON)
{
    __shared__ float shared_image[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float shared_grad_x[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float shared_grad_y[BLOCK_SIZE][BLOCK_SIZE];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= 1 && x < WIDTH - 1 && y >= 1 && y < HEIGHT - 1)
    {
        shared_image[threadIdx.y][threadIdx.x] = image[y * WIDTH + x];
        shared_grad_x[threadIdx.y][threadIdx.x] = grad_x[y * WIDTH + x];
        shared_grad_y[threadIdx.y][threadIdx.x] = grad_y[y * WIDTH + x];
        __syncthreads();

        float dx = shared_grad_x[threadIdx.y][threadIdx.x];
        float dy = shared_grad_y[threadIdx.y][threadIdx.x];
        float norm = sqrt(dx * dx + dy * dy + EPSILON);
        image[y * WIDTH + x] -= LAMBDA * (dx / norm + dy / norm);
    }
}

int main(int argc, char** argv)
{
    
    int WIDTH = atoi(argv[1]);
    int HEIGHT = atoi(argv[2]);

    float LAMBDA = atoi(argv[3]);
    float EPSILON = atoi(argv[4]);
    int MAX_ITERATIONS = atoi(argv[5]);

    // Initialize host memory for noisy image and denoised image
    float* host_image_noisy = (float*)malloc(WIDTH * HEIGHT * sizeof(float));
    float* host_image_denoised = (float*)malloc(WIDTH * HEIGHT * sizeof(float));

    // Initialize device memory for noisy image, denoised image, and gradients
    float* device_image_noisy, * device_image_denoised, * device_grad_x, * device_grad_y;
    cudaMalloc((void**)&device_image_noisy, WIDTH * HEIGHT * sizeof(float));
    cudaMalloc((void**)&device_image_denoised, WIDTH * HEIGHT * sizeof(float));
    cudaMalloc((void**)&device_grad_x, WIDTH * HEIGHT * sizeof(float));
    cudaMalloc((void**)&device_grad_y, WIDTH * HEIGHT * sizeof(float));

    // Generate random values for noisy image
    srand(time(NULL));
    for (int i = 0; i < WIDTH * HEIGHT; i++)
    {
    host_image_noisy[i] = (float)rand() / (float)RAND_MAX;
    }

    // Copy the noisy image from host to device
    cudaMemcpy(device_image_noisy, host_image_noisy, WIDTH * HEIGHT * sizeof(float), cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 dim_grid((WIDTH - 1) / BLOCK_SIZE + 1, (HEIGHT - 1) / BLOCK_SIZE + 1, 1);
    dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE, 1);

    // Perform TV regularization on the noisy image to obtain denoised image
    for (int i = 0; i < MAX_ITERATIONS; i++)
    {
        gradient<<<dim_grid, dim_block>>>(device_image_noisy, device_grad_x, device_grad_y, WIDTH, HEIGHT);
        update<<<dim_grid, dim_block>>>(device_image_denoised, device_grad_x, device_grad_y, WIDTH, HEIGHT, LAMBDA, EPSILON);


        // Swap noisy and denoised image pointers
        float* temp = device_image_noisy;
        device_image_noisy = device_image_denoised;
        device_image_denoised = temp;

    }


    // Copy the denoised image from device to host
    cudaMemcpy(host_image_denoised, device_image_denoised, WIDTH * HEIGHT * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(device_image_noisy);
    cudaFree(device_image_denoised);
    cudaFree(device_grad_x);
    cudaFree(device_grad_y);

    // Free host memory
    free(host_image_noisy);
    free(host_image_denoised);

    return 0;
}