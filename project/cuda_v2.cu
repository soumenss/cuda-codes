#include <cuda_runtime.h>
#include <fstream>

__global__ void median_filter_kernel(const uchar* input, uchar* output, int width, int height, int kernel_size)
{
    int padding_size = kernel_size / 2;
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int x = index % width;
    int y = index / width;
    int input_index, output_index;
    uchar neighborhood[kernel_size * kernel_size];

    if (x >= padding_size && x < width - padding_size && y >= padding_size && y < height - padding_size)
    {
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
                    uchar temp = neighborhood[i];
                    neighborhood[i] = neighborhood[j];
                    neighborhood[j] = temp;
                }
            }
        }

        // Write the output
        output_index = (y - padding_size) * (width - kernel_size + 1) + (x - padding_size);
        output[output_index] = neighborhood[kernel_size * kernel_size / 2];
    }
}

void median_filter(const uchar* input, uchar* output, int width, int height, int kernel_size)
{
    int padding_size = kernel_size / 2;
    int output_width = width - kernel_size + 1;
    int output_height = height - kernel_size + 1;
    size_t input_size = width * height * sizeof(uchar);
    size_t output_size = output_width * output_height * sizeof(uchar);

    // Allocate memory on the device
    uchar* d_input, * d_output;
    cudaMalloc(&d_input, input_size);
    cudaMalloc(&d_output, output_size);

    // Copy the input to the device
    cudaMemcpy(d_input, input, input_size, cudaMemcpyHostToDevice);

    // Define the grid and block sizes
    dim3 block_size(32, 32);
    dim3 grid_size((width * height + block_size.x * block_size.y - 1) / (block_size.x * block_size.y));

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
    std::ifstream file("image.raw", std::ios::binary);
    if (!file)
    {
        std::cerr << "Error: Could not open image file." << std::endl;
        return 1;
    }
    int width = 512;
    int height = 512;
    uchar input[width * height];
    file.read(reinterpret_cast<char*>(input), width * height);
    file.close();

    // Apply the median filter with a kernel size of

    int kernel_size = 5;
    uchar output[(width - kernel_size + 1) * (height - kernel_size + 1)];
    median_filter(input, output, width, height, kernel_size);

    // Write the output image
    std::ofstream output_file("output.raw", std::ios::binary);
    output_file.write(reinterpret_cast<const char*>(output), (width - kernel_size + 1) * (height - kernel_size + 1));
    output_file.close();

    return 0;

}