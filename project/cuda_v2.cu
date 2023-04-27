#include <cuda_runtime.h>
#include <fstream>
#include <iostream>

#define BLOCK_SIZE 32

__global__ void median_filter_kernel(const unsigned char* input, unsigned char* output, int width, int height, int kernel_size)
{
    int padding_size = kernel_size / 2;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= padding_size && x < width - padding_size && y >= padding_size && y < height - padding_size)
    {
        // unsigned char neighborhood[kernel_size * kernel_size];
        unsigned char* neighborhood = new unsigned char[kernel_size * kernel_size];

        int input_index, output_index = (y - padding_size) * (width - kernel_size + 1) + (x - padding_size);

        // Get the neighborhood
        for (int i = -padding_size; i <= padding_size; i++)
        {
            for (int j = -padding_size; j <= padding_size; j++)
            {
                input_index = (y + i) * width + x + j;
                neighborhood[(i + padding_size) * kernel_size + j + padding_size] = input[input_index];
            }
        }

        // Apply the median filter
        for (int i = 0; i < kernel_size * kernel_size; i++)
        {
            for (int j = i + 1; j < kernel_size * kernel_size; j++)
            {
                if (neighborhood[i] > neighborhood[j])
                {
                    unsigned char temp = neighborhood[i];
                    neighborhood[i] = neighborhood[j];
                    neighborhood[j] = temp;
                }
            }
        }

        // Write the output
        output[output_index] = neighborhood[kernel_size * kernel_size / 2];
    }

    delete[] neighborhood;
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
