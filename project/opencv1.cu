#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16
#define LAMBDA 0.1f
#define ITERATIONS 100

__global__ void tv_denoise_kernel(float* input, float* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    int index = y * width + x;

    float dx, dy;
    if (x == 0) {
        dx = input[index + 1] - input[index];
    } else if (x == width - 1) {
        dx = input[index] - input[index - 1];
    } else {
        dx = input[index + 1] - input[index] + input[index] - input[index - 1];
    }

    if (y == 0) {
        dy = input[index + width] - input[index];
    } else if (y == height - 1) {
        dy = input[index] - input[index - width];
    } else {
        dy = input[index + width] - input[index] + input[index] - input[index - width];
    }

    output[index] = input[index] - LAMBDA * (dx + dy);
}

int main() {
    cv::Mat image = cv::imread("noisy_image.jpg", cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        fprintf(stderr, "Failed to read image\n");
        return 1;
    }

    int width = image.cols;
    int height = image.rows;

    float* input = new float[width * height];
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            input[y * width + x] = static_cast<float>(image.at<uchar>(y, x)) / 255.0f;
        }
    }

    float* d_input;
    cudaMalloc(&d_input, width * height * sizeof(float));
    cudaMemcpy(d_input, input, width * height * sizeof(float), cudaMemcpyHostToDevice);

    float* d_output;
    cudaMalloc(&d_output, width * height * sizeof(float));

    dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_size((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    for (int i = 0; i < ITERATIONS; ++i) {
        tv_denoise_kernel<<<grid_size, block_size>>>(d_input, d_output, width, height);
        std::swap(d_input, d_output);
    }

    float* output = new float[width * height];
    cudaMemcpy(output, d_input, width * height * sizeof(float), cudaMemcpyDeviceToHost);

    cv::Mat denoised_image(height, width, CV_8UC1);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            denoised_image.at<uchar>(y, x) = static_cast<uchar>(output[y * width + x] * 255.0f);
        }
    }

    cv::imwrite("denoised_image.jpg", denoised_image);

    delete[] input;
    delete[] output;
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;

}