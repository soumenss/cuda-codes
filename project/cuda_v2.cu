#include <cuda_runtime.h>
#include <fstream>
#include <iostream>

#define BLOCK_SIZE 32

void median_filter(const unsigned char* input, unsigned char* output, int width, int height, int kernel_size)
{
    int padding_size = kernel_size / 2;

    // Add padding to the input image
    unsigned char* padded_input = new unsigned char[(width + 2 * padding_size) * (height + 2 * padding_size)];
    for (int i = 0; i < height + 2 * padding_size; i++)
    {
        for (int j = 0; j < width + 2 * padding_size; j++)
        {
            if (i < padding_size || i >= height + padding_size || j < padding_size || j >= width + padding_size)
            {
                padded_input[i * (width + 2 * padding_size) + j] = 0;
            }
            else
            {
                padded_input[i * (width + 2 * padding_size) + j] = input[(i - padding_size) * width + (j - padding_size)];
            }
        }
    }

    // Apply the median filter
    for (int i = 0; i < height - kernel_size + 1; i++)
    {
        for (int j = 0; j < width - kernel_size + 1; j++)
        {
            // Extract the neighborhood
            unsigned char* neighborhood = new unsigned char[kernel_size * kernel_size];
            for (int ii = 0; ii < kernel_size; ii++)
            {
                for (int jj = 0; jj < kernel_size; jj++)
                {
                    neighborhood[ii * kernel_size + jj] = padded_input[(i + ii) * (width + 2 * padding_size) + (j + jj)];
                }
            }

            // Compute the median of the neighborhood
            output[i * (width - kernel_size + 1) + j] = median(neighborhood, kernel_size * kernel_size);

            // Free memory allocated for neighborhood
            delete[] neighborhood;
        }
    }

    // Free memory allocated for padded input
    delete[] padded_input;
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
