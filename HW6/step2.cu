#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "cuda.h"
#include <time.h>

#define BLOCK_SIZE 32
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))
double start_time, elapsed_time;
double total_time = 0;

__global__ void sgemm_shared_mem_block(int M, int N, int K, float alpha, const float* A,
	const float* B, float beta, float* C) {
	extern __shared__ float As[BLOCK_SIZE * BLOCK_SIZE];
	extern __shared__ float Bs[BLOCK_SIZE * BLOCK_SIZE];

	// Compute the row and column indices for the current thread
  	const int threadCol = threadIdx.x % BLOCK_SIZE;
  	const int threadRow = threadIdx.x / BLOCK_SIZE;

	// Compute the block row and column indices
	const int cRow = blockIdx.x;
	const int cCol = blockIdx.y;

	// Advance pointers to the starting positions
	A += cRow * BLOCK_SIZE * K;    // row=cRow, col=0
	B += cCol * BLOCK_SIZE;        // row=0, col=cCol
	C += cRow * BLOCK_SIZE * N + cCol * BLOCK_SIZE; // row=cRow, col=cCol

	float tmp = 0.0;
	// The outer loop advances A along the columns and B along the rows
	// until we have fully calculated the result in C.
	for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCK_SIZE) {
		// Have each thread load one of the elements in A & B from
		// global memory into shared memory.
		// Make the threadCol (=threadIdx.x) the consecutive index
		// to allow global memory access coalescing
		As[threadRow * BLOCK_SIZE + threadCol] = A[threadRow * K + threadCol];
		Bs[threadRow * BLOCK_SIZE + threadCol] = B[threadRow * N + threadCol];

		// Block threads in this block until cache is fully populated
		__syncthreads();

		// Advance pointers onto the next chunk
		A += BLOCK_SIZE;
		B += BLOCK_SIZE * N;

		// Execute the dotproduct on the currently cached block
		for (int dotIdx = 0; dotIdx < BLOCK_SIZE; ++dotIdx) {
			tmp += As[threadRow * BLOCK_SIZE + dotIdx] *
				Bs[dotIdx * BLOCK_SIZE + threadCol];
		}
		// Need to sync again at the end, to avoid faster threads
		// fetching the next block into the cache before slower threads are done
		__syncthreads();
	}

	C[threadRow * N + threadCol] =
		alpha * tmp + beta * C[threadRow * N + threadCol];
}

int main() {
	const int M = 2048, N = 2048, K = 2048;
	const float alpha = 1.0f, beta = 0.0f;

	// Allocate memory on the host
	float* h_A = (float*)malloc(M * K * sizeof(float));
	float* h_B = (float*)malloc(K * N * sizeof(float));
	float* h_C = (float*)malloc(M * N * sizeof(float));

	// Initialize input matrices
	for (int i = 0; i < M * K; i++) {
		h_A[i] = (float)rand() / RAND_MAX;
	}
	for (int i = 0; i < K * N; i++) {
		h_B[i] = (float)rand() / RAND_MAX;
	}
	for (int i = 0; i < M * N; i++) {
		h_C[i] = (float)rand() / RAND_MAX;
	}

	// Allocate memory on the device
	float* d_A, *d_B, *d_C;
	for (int i = 0; i < 5; i++) {
		
		cudaMalloc(&d_A, M * K * sizeof(float));
		cudaMalloc(&d_B, K * N * sizeof(float));
		cudaMalloc(&d_C, M * N * sizeof(float));

		// Copy input matrices to the device
		cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_C, h_C, M * N * sizeof(float), cudaMemcpyHostToDevice);

		// Launch the kernel
		dim3 blockDim(32, 32, 1);
		dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32), 1);
		start_time = clock();
		size_t sharedMemSize = 2 * BLOCK_SIZE * BLOCK_SIZE * sizeof(float);

		sgemm_shared_mem_block <<<gridDim, blockDim, sharedMemSize >>> (M, N, K, alpha, d_A, d_B, beta, d_C);
		cudaDeviceSynchronize();

		elapsed_time = clock() - start_time;
		total_time += elapsed_time;

		// Copy the result back to the host
		cudaMemcpy(h_A, d_A, M * K * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_B, d_B, K * N * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

		// Free memory on the device
		cudaFree(d_A);
		cudaFree(d_B);
		cudaFree(d_C);


	}
	printf("Averaged Elapsed Time: %.20f\n", total_time / (5 * CLOCKS_PER_SEC));

	// print Results
	//for (int j = 0; j < K * M; j++)
	//{
	//	printf("%f \t", h_A[j]);
	//}

	//for (int j = 0; j < K * N; j++)
	//{
	//	printf("%f \t", h_B[j]);
	//}

	//for (int j = 0; j < M * N; j++)
	//{
	//	printf("%f \t", h_C[j]);
	//}

	// Free memory on the host
	free(h_A);
	free(h_B);
	free(h_C);

	return 0;
}