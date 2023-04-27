#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>

__global__ void median_filter_kernel(unsigned char* input, unsigned char* output, int width, int height, int window_size)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int radius = window_size / 2;
    int index = y * width + x;

    if (x < width && y < height)
    {
        std::vector<unsigned char> window;
        for (int i = -radius; i <= radius; ++i)
        {
            for (int j = -radius; j <= radius; ++j)
            {
                int nx = x + j;
                int ny = y + i;
                if (nx >= 0 && ny >= 0 && nx < width && ny < height)
                {
                    window.push_back(input[ny * width + nx]);
                }
            }
        }
        std::sort(window.begin(), window.end());
        output[index] = window[window.size() / 2];
    }
}

int main()
{
    const int width = 512;
    const int height = 512;
    const int window_size = 3;

    // Load input image from file
    std::ifstream file("input.jpg", std::ios::binary);
    std::vector<unsigned char> input_data(width * height);
    file.read(reinterpret_cast<char*>(input_data.data()), input_data.size());
    file.close();

    // Allocate memory on the device
    unsigned char* input_device;
    unsigned char* output_device;
    cudaMalloc(&input_device, input_data.size());
    cudaMalloc(&output_device, input_data.size());

    // Copy input data to device memory
    cudaMemcpy(input_device, input_data.data(), input_data.size(), cudaMemcpyHostToDevice);

    // Define block and grid sizes
    dim3 block_size(32, 32);
    dim3 grid_size((width + block_size.x - 1) / block_size.x, (height + block_size.y - 1) / block_size.y);

    // Launch the kernel
    median_filter_kernel<<<grid_size, block_size>>>(input_device, output_device, width, height, window_size);

    // Copy output data from device memory
    std::vector<unsigned char> output_data(width * height);
    cudaMemcpy(output_data.data(), output_device, output_data.size(), cudaMemcpyDeviceToHost);

    // Save output image to file
    std::ofstream out_file("output.jpg", std::ios::binary);
    out_file.write(reinterpret_cast<char*>(output_data.data()), output_data.size());
    out_file.close();

    // Free device memory
    cudaFree(input_device);
    cudaFree(output_device);

    return 0;
}
