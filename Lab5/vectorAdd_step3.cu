/*
*	In His Exalted Name
*	Vector Addition - Sequential Code
*	Ahmad Siavashi, Email: siavashi@aut.ac.ir
*	21/05/2018
*/

#define RUN_COUNT 10

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


void fillVector(int * v, size_t n);
void addVector(int * a, int *b, int *c, size_t n);
void printVector(int * v, size_t n);
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);


double start_time, elapsed_time;
double time_sum = 0;

//__global__ void addKernel(int *c, const int *a, const int *b)
//{
//	int i = threadIdx.x;
//	c[i] = a[i] + b[i];
//}

__global__ void addKernel(int *c, const int *a, const int *b, const int vectorSize, const int elements_per_thread)
{
	int start = (blockIdx.x * blockDim.x + threadIdx.x) * elements_per_thread;
	for (int i = start; i - start < elements_per_thread && (i < vectorSize); i++) {
		c[i] = a[i] + b[i];
	}
}

int main()
{
	//const int vectorSize = 1024;
	const int vectorSize = 1 << 26;

	//int a[vectorSize], b[vectorSize], c[vectorSize];

	int *a = (int *)malloc(sizeof(int) * vectorSize);
	int *b = (int *)malloc(sizeof(int) * vectorSize);
	int *c = (int *)malloc(sizeof(int) * vectorSize);

	fillVector(a, vectorSize);
	fillVector(b, vectorSize);

	for (int i = 0; i < RUN_COUNT; i++){
		addWithCuda(c, a, b, vectorSize);
		//addVector(a, b, c, vectorSize);
		time_sum += elapsed_time;
	}


	//printVector(c, vectorSize);

	printf("Averaged Elapsed Time : %.20f", time_sum / (RUN_COUNT * CLOCKS_PER_SEC));

	return EXIT_SUCCESS;
}

// Fills a vector with data
void fillVector(int * v, size_t n) {
	int i;
	for (i = 0; i < n; i++) {
		v[i] = i;
	}
}

// Adds two vectors
void addVector(int * a, int *b, int *c, size_t n) {
	start_time = clock();
	int i;
	for (i = 0; i < n; i++) {
		c[i] = a[i] + b[i];
	}
	elapsed_time = clock() - start_time;

}

// Prints a vector to the stdout.
void printVector(int * v, size_t n) {
	int i;
	printf("[-] Vector elements: ");
	for (i = 0; i < n; i++) {
		printf("%d, ", v[i]);
	}
	printf("\b\b  \n");
}

// 
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size) {
	int *dev_a = 0;
	int *dev_b = 0;
	int *dev_c = 0;
	cudaError_t cudaStatus;

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		printf("cudaSetDevice failed! Do you have a CUDA-capable GPU installed?");
	}

	cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		printf("cudaMalloc failed!");
	}

	cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		printf("cudaMalloc failed!");
	}

	cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		printf("cudaMalloc failed!");
	}

	cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		printf("cudaMemcpy failed!");
	}

	cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		printf("cudaMemcpy failed!");
	}

	//int ELEMENTS_PER_THREAD = (1 << 26) /1024;
	int ELEMENTS_PER_THREAD = 1;
	dim3 NUM_THREADS(1024, 1, 1);	// Threads per block
	dim3 NUM_BLOCKS((size + (NUM_THREADS.x * ELEMENTS_PER_THREAD) - 1) / (NUM_THREADS.x * ELEMENTS_PER_THREAD), 1, 1);

	printf("elements per thread: %d, threads per blocks: %d, blocks: %d\n", ELEMENTS_PER_THREAD, NUM_THREADS.x, NUM_BLOCKS.x);

	start_time = clock();
	//addKernel <<< 1, 1024 >>> (dev_c, dev_a, dev_b);
	addKernel << <NUM_BLOCKS, NUM_THREADS >> > (dev_c, dev_a, dev_b, size, ELEMENTS_PER_THREAD);


	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		printf("addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		printf("cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
	}
	elapsed_time = clock() - start_time;

	//printf("\n %f\t", elapsed_time);

	cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		printf("cudaMemcpy failed!");
	}


	cudaFree(dev_c);
	cudaFree(dev_a);
	cudaFree(dev_b);

	return cudaStatus;
}












