/* Includes, system */
#include <cstdio>
#include <iostream>
#include <vector>
#include <numeric>
#include <chrono>
#include <cstdio>
#include <stdio.h>
/* Includes, cuda */
#include <cuda_runtime.h>
#include <cudnn.h>
#include <cublas_v2.h>
#include <ctime>

struct Tensor4d
{
    cudnnTensorDescriptor_t desc;
    void *data;
    size_t data_size;

    Tensor4d(int n, int c, int h, int w)
    {
        cudnnCreateTensorDescriptor(&desc);
        cudnnSetTensor4dDescriptor(desc,
                                   CUDNN_TENSOR_NCHW,
                                   CUDNN_DATA_FLOAT,
                                   n, c, h, w);
        data_size = n * c * h * w;
        cudaMalloc((void **)&data, data_size * sizeof(float));
    }
    ~Tensor4d()
    {
        cudaFree(data);
    }
};

struct Bias4d
{
    cudnnTensorDescriptor_t desc;
    void *data;
    size_t data_size;

    Bias4d(int n, int c, int h, int w)
    {
        cudnnCreateTensorDescriptor(&desc);
        cudnnSetTensor4dDescriptor(desc,
                                   CUDNN_TENSOR_NCHW,
                                   CUDNN_DATA_FLOAT,
                                   n, c, h, w);
        data_size = n * c * h * w;
        float *tmp = (float *)malloc(data_size * sizeof(float));
        for (int i = 0; i < data_size; i++)
        {
            tmp[i] = (float)std::rand() / RAND_MAX / 1000;
        }
        cudaMalloc((void **)&data, data_size * sizeof(float));
        auto code = cudaMemcpy(data, tmp, data_size * sizeof(float),
                               cudaMemcpyHostToDevice);
    }
    ~Bias4d()
    {
        cudaFree(data);
    }
};

struct Filter4d
{
    cudnnFilterDescriptor_t desc;
    void *data;
    size_t data_size;

    Filter4d(int n, int c, int h, int w)
    {
        cudnnCreateFilterDescriptor(&desc);
        cudnnSetFilter4dDescriptor(desc,
                                   CUDNN_DATA_FLOAT,
                                   CUDNN_TENSOR_NCHW,
                                   n, c, h, w);
        data_size = n * c * h * w;
        float *tmp = (float *)malloc(data_size * sizeof(float));
        for (int i = 0; i < data_size; i++)
        {
            tmp[i] = (float)std::rand() / RAND_MAX / 1000;
        }

        cudaMalloc((void **)&data, data_size * sizeof(float));
        auto code = cudaMemcpy(data, tmp, data_size * sizeof(float),
                               cudaMemcpyHostToDevice);
    }
    ~Filter4d()
    {
        cudaFree(data);
    }
};

struct zeros
{
    void *data;
    size_t data_size;
    zeros(std::vector<int> dims)
    {
        data_size = std::accumulate(dims.begin(),
                                    dims.end(),
                                    1,
                                    std::multiplies<int>());
        std::vector<float> host_data(data_size);
        for (int i = 0; i < data_size; i++)
            host_data[i] = 0;

        cudaMalloc((void **)&data, data_size * sizeof(float));
        cudaMemcpy(data, host_data.data(), data_size * sizeof(float),
                   cudaMemcpyHostToDevice);
    }
    ~zeros()
    {
        cudaFree(data);
    }
};

// Define the CUDA kernel for convolution
__global__ void conv2DKernel(
    const float* input, const float* filter,
    float* output, int w, int h, int c, int k,
    int filter_w, int filter_h, int pad_w, int pad_h,
    int stride_w, int stride_h, int out_w, int out_h) {

    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int out_z = blockIdx.z * blockDim.z + threadIdx.z;

    if (out_x < out_w && out_y < out_h && out_z < k) {
        float sum = 0.0f;
        for (int i = 0; i < c; ++i) {
            for (int j = 0; j < filter_h; ++j) {
                for (int l = 0; l < filter_w; ++l) {
                    int in_x = out_x * stride_w - pad_w + l;
                    int in_y = out_y * stride_h - pad_h + j;
                    if (in_x >= 0 && in_x < w && in_y >= 0 && in_y < h) {
                        sum += input[(i * h + in_y) * w + in_x] * filter[((out_z * c + i) * filter_h + j) * filter_w + l];
                    }
                }
            }
        }
        output[(out_z * out_h + out_y) * out_w + out_x] = sum;
    }
}

void cuConv2D(float *input, float *output, int w, int h, int c, int n, int k,
              int filter_w, int filter_h, int dilation_w, int dilation_h,
              int pad_w, int pad_h, int wstride, int hstride) {

    int out_w = (w + 2 * pad_w - dilation_w * (filter_w - 1) - 1) / wstride + 1;
    int out_h = (h + 2 * pad_h - dilation_h * (filter_h - 1) - 1) / hstride + 1;

    float *h_filter = (float *)malloc(k * c * filter_w * filter_h * sizeof(float));
    for (int i = 0; i < k * c * filter_w * filter_h; i++) {
        h_filter[i] = (float)std::rand() / RAND_MAX / 1000;
    }

    // Allocate GPU memory
    float *d_input, *d_filter, *d_output;
    cudaMalloc(&d_input, n * c * h * w * sizeof(float));
    cudaMalloc(&d_filter, k * c * filter_w * filter_h * sizeof(float));
    cudaMalloc(&d_output, n * k * out_h * out_w * sizeof(float));

    cudaMemcpy(d_input, input, n * c * h * w * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, h_filter, k * c * filter_w * filter_h * sizeof(float), cudaMemcpyHostToDevice);

    // Define block and grid sizes
    int blockSize = 16;
    dim3 threadsPerBlock(blockSize, blockSize, 1);
    dim3 numBlocks((out_w + blockSize - 1) / blockSize, (out_h + blockSize - 1) / blockSize, k);

    auto start = std::chrono::steady_clock::now();

    // Launch the custom convolution kernel
    conv2DKernel<<<numBlocks, threadsPerBlock>>>(d_input, d_filter, d_output, w, h, c, k, filter_w, filter_h, pad_w, pad_h, wstride, hstride, out_w, out_h);
    cudaDeviceSynchronize();

    auto end = std::chrono::steady_clock::now();

    int fwd_time = static_cast<int>(std::chrono::duration<double, std::micro>(end - start).count());
    std::cout << " " << fwd_time << " ms" << std::endl;

    cudaMemcpy(output, d_output, n * k * out_h * out_w * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_filter);
    cudaFree(d_output);
    free(h_filter);
}


// Define the CUDA kernel for matrix multiplication
__global__ void matrixMulKernel(const float* A, const float* B, float* C, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        float value = 0.0f;
        for (int e = 0; e < k; ++e) {
            value += A[row * k + e] * B[e * n + col];
        }
        C[row * n + col] = value;
    }
}

void cuFC(float *input, float *output, int left, int right) {
    int m = 1, k = left, n = right;

    float *h_B = (float *)malloc(left * right * sizeof(float));
    for (int i = 0; i < left * right; i++) {
        h_B[i] = (float)std::rand() / RAND_MAX / 1000;
    }

    // Allocate 3 arrays on GPU
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, left * sizeof(float));
    cudaMalloc(&d_B, left * right * sizeof(float));
    cudaMalloc(&d_C, right * sizeof(float));

    cudaMemcpy(d_A, input, left * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, left * right * sizeof(float), cudaMemcpyHostToDevice);

    // Define block and grid sizes
    int blockSize = 16; // Adjust as necessary
    dim3 threadsPerBlock(blockSize, blockSize);
    dim3 numBlocks((n + blockSize - 1) / blockSize, (m + blockSize - 1) / blockSize);

    auto start = std::chrono::steady_clock::now();

    // Launch the custom matrix multiplication kernel
    matrixMulKernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, m, n, k);
    cudaDeviceSynchronize();

    auto end = std::chrono::steady_clock::now();

    int fwd_time = static_cast<int>(std::chrono::duration<double, std::micro>(end - start).count());
    std::cout << " " << fwd_time << " ms" << std::endl;

    cudaMemcpy(output, d_C, right * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_B);
}

// Define the CUDA kernel for max pooling
__global__ void maxPoolKernel(const float *input, float *output, int w, int h, int c, int n, int pool_w, int pool_h, int stride_w, int stride_h) {
    int out_w = w / stride_w;
    int out_h = h / stride_h;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elems = n * c * out_h * out_w;

    if (idx < total_elems) {
        int out_x = idx % out_w;
        int out_y = (idx / out_w) % out_h;
        int channel = (idx / (out_w * out_h)) % c;
        int batch = idx / (out_w * out_h * c);

        int start_x = out_x * stride_w;
        int start_y = out_y * stride_h;

        // Shared memory for pooling
        extern __shared__ float shared_input[];
        float* shared_channel = &shared_input[threadIdx.z * pool_h * pool_w];

        // Load input data into shared memory
        for (int i = 0; i < pool_h; ++i) {
            for (int j = 0; j < pool_w; ++j) {
                int cur_x = start_x + j;
                int cur_y = start_y + i;
                int input_idx = ((batch * c + channel) * h + cur_y) * w + cur_x;
                shared_channel[i * pool_w + j] = (cur_x < w && cur_y < h) ? input[input_idx] : -FLT_MAX;
            }
        }

        __syncthreads();  // Ensure all threads have loaded data into shared memory

        // Compute max value within the pooling window
        float max_val = -FLT_MAX;
        for (int i = 0; i < pool_h; ++i) {
            for (int j = 0; j < pool_w; ++j) {
                max_val = fmaxf(max_val, shared_channel[i * pool_w + j]);
            }
        }

        // Store result in output
        int output_idx = ((batch * c + channel) * out_h + out_y) * out_w + out_x;
        output[output_idx] = max_val;
    }
}



void cuMaxPool(float *input, float *output, int w, int h, int c, int n) {
    const int pool_w = 2;
    const int pool_h = 2;
    const int stride_w = 2;
    const int stride_h = 2;

    int out_w = w / stride_w;
    int out_h = h / stride_h;

    float *d_input, *d_output;

    // Allocate memory on the device
    cudaMalloc(&d_input, n * c * h * w * sizeof(float));
    cudaMalloc(&d_output, n * c * out_h * out_w * sizeof(float));

    // Copy input data from host to device
    cudaMemcpy(d_input, input, n * c * h * w * sizeof(float), cudaMemcpyHostToDevice);

    // Launch the custom max pooling kernel
    int total_elems = n * c * out_h * out_w;
    int blockSize = 256;
    int numBlocks = (total_elems + blockSize - 1) / blockSize;

    auto start = std::chrono::steady_clock::now();
    maxPoolKernel<<<numBlocks, blockSize>>>(d_input, d_output, w, h, c, n, pool_w, pool_h, stride_w, stride_h);
    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();

    // Copy output data from device to host
    cudaMemcpy(output, d_output, n * c * out_h * out_w * sizeof(float), cudaMemcpyDeviceToHost);

    // Measure and print the elapsed time
    int fwd_time = static_cast<int>(std::chrono::duration<double, std::micro>(end - start).count());
    std::cout << " " << fwd_time << " ms" << std::endl;

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}

int main()
{
    std::srand(std::time(0));

    float *input;
    float *output;

    int data_size = 224 * 224 * 3 * 1;
    input = (float *)malloc(data_size * sizeof(float));
    for (int i = 0; i < data_size; i++)
    {
        input[i] = (float)std::rand() / RAND_MAX;
    }

    // ===============  1 =====================
    std::cout << "CONV 224x224x64";
    output = (float *)malloc(224 * 224 * 64 * 1 * sizeof(float));
    cuConv2D(input, output, 224, 224, 3, 1, 64, 3, 3, 1, 1, 1, 1, 1, 1);
    std::swap(input, output);
    free(output);

    std::cout << "CONV 224x224x64";
    output = (float *)malloc(224 * 224 * 64 * 1 * sizeof(float));
    cuConv2D(input, output, 224, 224, 64, 1, 64, 3, 3, 1, 1, 1, 1, 1, 1);
    std::swap(input, output);
    free(output);

    std::cout << "POOLMAX 112x112x64";
    output = (float *)malloc(112 * 112 * 64 * sizeof(float));
    cuMaxPool(input, output, 224, 224, 64, 1);
    std::swap(input, output);
    free(output);

    // ===============  2 =====================
    std::cout << "CONV 112x112x128";
    output = (float *)malloc(112 * 112 * 128 * 1 * sizeof(float));
    cuConv2D(input, output, 112, 112, 64, 1, 64, 3, 3, 1, 1, 1, 1, 1, 1);
    std::swap(input, output);
    free(output);

    std::cout << "CONV 112x112x128";
    output = (float *)malloc(112 * 112 * 128 * 1 * sizeof(float));
    cuConv2D(input, output, 112, 112, 128, 1, 64, 3, 3, 1, 1, 1, 1, 1, 1);
    std::swap(input, output);
    free(output);

    std::cout << "POOLMAX 56x56x128";
    output = (float *)malloc(56 * 56 * 128 * sizeof(float));
    cuMaxPool(input, output, 112, 112, 128, 1);
    std::swap(input, output);
    free(output);

    // ===============  3 =====================
    std::cout << "CONV 56x56x256";
    output = (float *)malloc(56 * 56 * 256 * 1 * sizeof(float));
    cuConv2D(input, output, 56, 56, 128, 1, 64, 3, 3, 1, 1, 1, 1, 1, 1);
    std::swap(input, output);
    free(output);

    std::cout << "CONV 56x56x256";
    output = (float *)malloc(56 * 56 * 256 * 1 * sizeof(float));
    cuConv2D(input, output, 56, 56, 256, 1, 64, 3, 3, 1, 1, 1, 1, 1, 1);
    std::swap(input, output);
    free(output);

    std::cout << "POOLMAX 28x28x256";
    output = (float *)malloc(28 * 28 * 256 * sizeof(float));
    cuMaxPool(input, output, 56, 56, 256, 1);
    std::swap(input, output);
    free(output);

    // ===============  4 =====================
    std::cout << "CONV 28x28x512";
    output = (float *)malloc(28 * 28 * 512 * 1 * sizeof(float));
    cuConv2D(input, output, 28, 28, 256, 1, 64, 3, 3, 1, 1, 1, 1, 1, 1);
    std::swap(input, output);
    free(output);

    std::cout << "CONV 28x28x512";
    output = (float *)malloc(28 * 28 * 512 * 1 * sizeof(float));
    cuConv2D(input, output, 28, 28, 512, 1, 64, 3, 3, 1, 1, 1, 1, 1, 1);
    std::swap(input, output);
    free(output);

    std::cout << "POOLMAX 14x14x512";
    output = (float *)malloc(14 * 14 * 512 * sizeof(float));
    cuMaxPool(input, output, 28, 28, 512, 1);
    std::swap(input, output);
    free(output);

    // ===============  5 =====================
    std::cout << "CONV 14x14x1024";
    output = (float *)malloc(14 * 14 * 1024 * 1 * sizeof(float));
    cuConv2D(input, output, 14, 14, 512, 1, 64, 3, 3, 1, 1, 1, 1, 1, 1);
    std::swap(input, output);
    free(output);

    std::cout << "CONV 14x14x1024";
    output = (float *)malloc(14 * 14 * 1024 * 1 * sizeof(float));
    cuConv2D(input, output, 14, 14, 1024, 1, 64, 3, 3, 1, 1, 1, 1, 1, 1);
    std::swap(input, output);
    free(output);

    std::cout << "POOLMAX 7x7x1024";
    output = (float *)malloc(7 * 7 * 1024 * sizeof(float));
    cuMaxPool(input, output, 14, 14, 1024, 1);
    std::swap(input, output);
    free(output);

    // ===============  6 =====================
    std::cout << "FC 4096";
    output = (float *)malloc(4096 * sizeof(float));
    cuFC(input, output, 7 * 7 * 1024, 4096);
    std::swap(input, output);
    free(output);

    std::cout << "FC 4096";
    output = (float *)malloc(4096 * sizeof(float));
    cuFC(input, output, 4096, 4096);
    std::swap(input, output);
    free(output);

    std::cout << "FC 1000";
    output = (float *)malloc(1000 * sizeof(float));
    cuFC(input, output, 4096, 1000);
    
    free(input);
    free(output);
}