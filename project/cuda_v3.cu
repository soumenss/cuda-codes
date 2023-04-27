#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>

#include <cuda_runtime.h>

using namespace std;

__global__ void medianFilter(const unsigned char* src, unsigned char* dst, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= 1 && y >= 1 && x < width - 1 && y < height - 1) {
        vector<unsigned char> window;
        for (int i = -1; i <= 1; ++i) {
            for (int j = -1; j <= 1; ++j) {
                window.push_back(src[(y + i) * width + x + j]);
            }
        }
        sort(window.begin(), window.end());
        dst[y * width + x] = window[4];
    } else {
        dst[y * width + x] = src[y * width + x];
    }
}

int main() {
    // Load the image file using standard C++ file I/O operations
    ifstream inFile("noisy_image.raw", ios::binary);
    if (!inFile.is_open()) {
        cerr << "Failed to open file." << endl;
        return 1;
    }
    const int width = 512;
    const int height = 512;
    vector<unsigned char> inputImage(width * height);
    inFile.read(reinterpret_cast<char*>(inputImage.data()), inputImage.size());
    inFile.close();

    // Allocate memory on the GPU
    unsigned char* devInputImage = nullptr;
    unsigned char* devOutputImage = nullptr;
    cudaMalloc(&devInputImage, inputImage.size());
    cudaMalloc(&devOutputImage, inputImage.size());

    // Copy the input image to the GPU
    cudaMemcpy(devInputImage, inputImage.data(), inputImage.size(), cudaMemcpyHostToDevice);

    // Compute the block and grid dimensions
    const dim3 blockDim(16, 16);
    const dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

    // Perform the median filter operation
    medianFilter<<<gridDim, blockDim>>>(devInputImage, devOutputImage, width, height);

    // Copy the output image from the GPU
    vector<unsigned char> outputImage(width * height);
    cudaMemcpy(outputImage.data(), devOutputImage, outputImage.size(), cudaMemcpyDeviceToHost);

    // Save the denoised image to a file
    ofstream outFile("denoised_image.raw", ios::binary);
    outFile.write(reinterpret_cast<const char*>(outputImage.data()), outputImage.size());
    outFile.close();

    // Free the GPU memory
    cudaFree(devInputImage);
    cudaFree(devOutputImage);

    return 0;
}
