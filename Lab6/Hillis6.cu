#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#define RUN_COUNT 5
double start_time, elapsed_time;

__global__ void prefixSumShared(int *data, int *prefixSum, int n) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    extern __shared__ int sharedMem[];

    // Load data into shared memory
    if (index < n) {
        sharedMem[threadIdx.x] = data[index];
    } else {
        sharedMem[threadIdx.x] = 0;
    }
    __syncthreads();

    // Perform parallel reduction in shared memory
    for (int d = 1; d < blockDim.x; d *= 2) {
        if (threadIdx.x >= d) {
            sharedMem[threadIdx.x] += sharedMem[threadIdx.x - d];
        }
        __syncthreads();
    }

    // Write the computed prefix sum to the output array
    if (index < n) {
        prefixSum[index] = sharedMem[threadIdx.x];
    }
}

__global__ void prefixSumNoShared(int* data, int* prefixSum, int n) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n) return;

    // Initialize the prefix sum with the first element
    prefixSum[index] = data[index];
    __syncthreads();

    for (int d = 1; d < n; d <<= 1) {
        int temp = 0;
        if (index >= d) {
            temp = prefixSum[index - d];
        }
        __syncthreads();

        if (index >= d) {
            prefixSum[index] += temp;
        }
        __syncthreads();
    }
}

int main() {
    int n;
    printf("Enter the length of the input array: ");
    scanf("%d", &n);

    // Allocate memory on host for data and prefix sum
    int *data = (int*)malloc(n * sizeof(int));
    int *prefixSum = (int*)malloc(n * sizeof(int));

    // Get input data from user
    for (int i = 0; i < n; i++) {
        data[i] = i;
    }

    // Choose between using shared memory or not
    int useSharedMemory;
    printf("Use shared memory (1) or not (0)? ");
    scanf("%d", &useSharedMemory);

    int blockSize = 32; // Adjust block size as needed
    int threadsPerBlock = (n + blockSize - 1) / blockSize;

    // Allocate memory on device for data and prefix sum
    int *d_data, *d_prefixSum;
    cudaMalloc(&d_data, n * sizeof(int));
    cudaMalloc(&d_prefixSum, n * sizeof(int));

    // Copy data to device
    cudaMemcpy(d_data, data, n * sizeof(int), cudaMemcpyHostToDevice);

	double total_time = 0;

    // Launch the appropriate kernel
    if (useSharedMemory) {
		for(int i = 0; i < RUN_COUNT; i++){
			start_time = clock();
			prefixSumShared<<<threadsPerBlock, blockSize, blockSize*sizeof(int)>>>(d_data, d_prefixSum, n);
			 // Synchronize the device to ensure completion
			cudaDeviceSynchronize();
			elapsed_time = clock() - start_time;
			total_time += elapsed_time;
		}
	}
    else {
		for(int i = 0; i < RUN_COUNT; i++){
			start_time = clock();
			prefixSumNoShared<<<threadsPerBlock, blockSize>>>(d_data, d_prefixSum, n);
			// Synchronize the device to ensure completion
			cudaDeviceSynchronize();
			elapsed_time = clock() - start_time;
			total_time += elapsed_time;
		}
    }

	// Check for errors again
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        // Handle error...
    }

    // Copy prefix sum results back from device
    cudaMemcpy(prefixSum, d_prefixSum, n * sizeof(int), cudaMemcpyDeviceToHost);

	// Check for errors again
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        // Handle error...
    }

    // Print the prefix sum
	printf("%f\n", total_time/(CLOCKS_PER_SEC*RUN_COUNT));
    printf("Prefix sum: ");
    for (int i = 0; i < n; i++) {
		printf("%d\t", prefixSum[i]);
	}
}