#include <iostream>
#include <fstream>
#include <cmath>

// Define the image dimensions
const int WIDTH = 512;
const int HEIGHT = 512;
const int SIZE = WIDTH * HEIGHT;

// Define the regularization parameters
const float LAMBDA = 0.05f;
const float THETA = 0.125f;

// Define the TV regularization function
__device__ float tv_func(float a, float b)
{
    float val = sqrtf(a*a + b*b);
    return 1.0f / sqrtf(1.0f + THETA*val);
}

// Define the CUDA kernel
__global__ void denoise_kernel(float* img_in, float* img_out)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < WIDTH && y < HEIGHT)
    {
        // Compute the TV regularization term
        float reg_term = 0.0f;
        if (x > 0) reg_term += tv_func(img_in[y*WIDTH + x] - img_in[y*WIDTH + x-1], 0.0f);
        if (x < WIDTH-1) reg_term += tv_func(img_in[y*WIDTH + x] - img_in[y*WIDTH + x+1], 0.0f);
        if (y > 0) reg_term += tv_func(img_in[y*WIDTH + x] - img_in[(y-1)*WIDTH + x], 0.0f);
        if (y < HEIGHT-1) reg_term += tv_func(img_in[y*WIDTH + x] - img_in[(y+1)*WIDTH + x], 0.0f);

        // Compute the denoised pixel value
        img_out[y*WIDTH + x] = img_in[y*WIDTH + x] + LAMBDA*reg_term;
    }
}

int main()
{
    // Load the input image from file
    std::ifstream file("input.jpg", std::ios::binary);
    char* img_data = new char[SIZE];
    file.read(img_data, SIZE);
    file.close();

    // Normalize the pixel values
    float* img_in = new float[SIZE];
    for (int i = 0; i < SIZE; i++)
    {
        int val = (unsigned char)img_data[i];
        img_in[i] = val / 255.0f;
    }
    delete[] img_data;

    // Allocate device memory
    float* d_img_in, *d_img_out;
    cudaMalloc((void**)&d_img_in, SIZE*sizeof(float));
    cudaMalloc((void**)&d_img_out, SIZE*sizeof(float));

    // Copy host memory to device
    cudaMemcpy(d_img_in, img_in, SIZE*sizeof(float), cudaMemcpyHostToDevice);

    // Initialize thread block and kernel grid dimensions
    dim3 block_dim(16, 16, 1);
    dim3 grid_dim((WIDTH + block_dim.x - 1) / block_dim.x, (HEIGHT + block_dim.y - 1) / block_dim.y, 1);

    // Invoke CUDA kernel
    denoise_kernel<<<grid_dim, block_dim>>>(d_img_in, d_img_out);

    // Copy results from device to host
    float* img_out = new float[SIZE];
    cudaMemcpy(img_out, d_img_out, SIZE*sizeof(float), cudaMemcpyDeviceToHost);

    // Deallocate device memory
    cudaFree(d_img_in);
    cudaFree(d_img_out);

    // Save the denoised image to file
    std::ofstream outfile("output.jpg", std::ios::binary);
    char* out_data = new char[SIZE];
    for (int i = 0; i < SIZE; i++)
    {
        int val = (int)(img_out[i] * 255.0f + 0.5f);
        if (val > 255) val = 255;
        if (val < 0) val = 0;
        out_data[i] = (char)val;
    }
    outfile.write(out_data, SIZE);
    outfile.close();
    delete[] out_data;

    // Deallocate host memory
    delete[] img_in;
    delete[] img_out;

    return 0;

}