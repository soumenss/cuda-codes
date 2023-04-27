#include <cuda_runtime.h>
#include <fstream>
#include <iostream>

#include <thrust/device_vector.h>
#include <thrust/sort.h>

#define BLOCK_SIZE 32

__device__ unsigned char median(unsigned char* neighborhood, int size)
{
    thrust::sort(thrust::device, neighborhood, neighborhood + size);
    return neighborhood[size / 2];
}

__global__ void median_filter_kernel(const unsigned char* input, unsigned char* output, int width, int height, int kernel_size)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= kernel_size / 2 && row < height - kernel_size / 2 && col >= kernel_size / 2 && col < width - kernel_size / 2)
    {
        // Allocate memory for the neighborhood on the device
        unsigned char* d_neighborhood;
        cudaMalloc(&d_neighborhood, kernel_size * kernel_size * sizeof(unsigned char));

        // Copy the neighborhood from the input to the device
        for (int i = 0; i < kernel_size; i++)
        {
            int input_row = row - kernel_size / 2 + i;
            for (int j = 0; j < kernel_size; j++)
            {
                int input_col = col - kernel_size / 2 + j;
                int input_index = input_row * width + input_col;
                int neighborhood_index = i * kernel_size + j;
                cudaMemcpy(&d_neighborhood[neighborhood_index], &input[input_index], sizeof(unsigned char), cudaMemcpyHostToDevice);
            }
        }

        // Compute the median of the neighborhood on the device
        output[row * width + col] = median(d_neighborhood, kernel_size * kernel_size);

        // Free the memory for the neighborhood on the device
        cudaFree(d_neighborhood);
    }
}


void median_filter(const unsigned char* input, unsigned char* output, int width, int height, int kernel_size)
{
    int padding_size = kernel_size / 2;
    int output_width = width - kernel_size + 1;
    int output_height = height - kernel_size + 1;
    size_t input_size = width * height * sizeof(unsigned char);
    size_t output_size = output_width * output_height * sizeof(unsigned char);

    // Allocate memory on the device
    unsigned char* d_input, * d_output;
    cudaMalloc(&d_input, input_size);
    cudaMalloc(&d_output, output_size);

    // Copy the input to the device
    cudaMemcpy(d_input, input, input_size, cudaMemcpyHostToDevice);

    // Define the grid and block sizes
    dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_size((output_width + BLOCK_SIZE - 1) / BLOCK_SIZE, (output_height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Call the kernel
    median_filter_kernel<<<grid_size, block_size>>>(d_input, d_output, width, height, kernel_size);

    // Copy the output to the host
    cudaMemcpy(output, d_output, output_size, cudaMemcpyDeviceToHost);

    // Free the memory on the device
    cudaFree(d_input);
    cudaFree(d_output);
}

int main()
{
    // Load the image
    std::ifstream file("input.jpg", std::ios::binary);
    if (!file)
    {
        std::cerr << "Error: Could not open image file." << std::endl;
        return 1;
    }

    int width = 512;
    int height = 512;
    unsigned char input[width * height];
    file.read(reinterpret_cast<char*>(input), width * height);
    file.close();

    // Apply the median filter with a kernel size of 5
    int kernel_size = 5;
    unsigned char output[(width - kernel_size + 1) * (height - kernel_size + 1)];
    median_filter(input, output, width, height, kernel_size);

    // Write the output image
    std::ofstream output_file("output.jpg", std::ios::binary);
    output_file.write(reinterpret_cast<const char*>(output), (width - kernel_size + 1) * (height - kernel_size + 1));
    output_file.close();

    return 0;
}
