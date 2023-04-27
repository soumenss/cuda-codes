#include <iostream>
#include <fstream>
#include <cuda_runtime.h>

// Define the kernel function for thresholding
__global__ void threshold(unsigned char* input, unsigned char* output, int threshold_value, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        int idx = y * width + x;
        if (input[idx] >= threshold_value) {
            output[idx] = 255;
        }
        else {
            output[idx] = 0;
        }
    }
}

int main() {
    // Load the image file using standard C++ file I/O operations
    std::ifstream file("input_image.png", std::ios::binary);
    if (!file) {
        std::cerr << "Error: Could not open input image file." << std::endl;
        return 1;
    }

    // Read the image dimensions and allocate memory for the image data on the CPU
    int width, height, max_val;
    file >> width >> height >> max_val;
    unsigned char* h_input_image = new unsigned char[width * height];
    unsigned char* h_output_image = new unsigned char[width * height];

    // Read the image data from the file and store it in the h_input_image buffer
    file.read((char*)h_input_image, width * height);

    // Copy the image data from the CPU to the GPU
    unsigned char* d_input_image;
    unsigned char* d_output_image;
    cudaMalloc(&d_input_image, width * height);
    cudaMalloc(&d_output_image, width * height);
    cudaMemcpy(d_input_image, h_input_image, width * height, cudaMemcpyHostToDevice);

    // Call the threshold kernel function
    dim3 block_size(32, 32);
    dim3 grid_size((width + block_size.x - 1) / block_size.x, (height + block_size.y - 1) / block_size.y);
    threshold<<<grid_size, block_size>>>(d_input_image, d_output_image, 128, width, height);

    // Copy the result from the GPU to the CPU
    cudaMemcpy(h_output_image, d_output_image, width * height, cudaMemcpyDeviceToHost);

    // Save the result to a new image file with a different name
    std::ofstream outfile("output_image.png", std::ios::binary);
    outfile << "P5\n" << width << " " << height << "\n" << max_val << "\n";
    outfile.write((char*)h_output_image, width * height);

    // Free memory
    delete[] h_input_image;
    delete[] h_output_image;
    cudaFree(d_input_image);
    cudaFree(d_output_image);

    return 0;
}
