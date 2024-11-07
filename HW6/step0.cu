
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "cuda.h"
#include <time.h>

#define BLOCK_SIZE 32
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))
double start_time, elapsed_time;
double total_time = 0;

__global__ void sgemm_naive(int M, int N, int K, float alpha, const float * A,
	const float * B, float beta, float * C) {
	// compute position in C that this thread is responsible for

	// step 0
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	// `if` condition is necessary for when M or N aren't multiples of 32.
	if (x < M && y < N) {
		float tmp = 0.0;
		for (int i = 0; i < K; ++i) {
			tmp += A[x * K + i] * B[i * N + y];
		}
		// C = α*(A@B)+β*C
		C[x * N + y] = alpha * tmp + beta * C[x * N + y];
	}
}

int main() {
	const int M = 2048, N = 2048, K = 2048;
	const float alpha = 1.0f, beta = 0.0f;

	// Allocate memory on the host
	float *h_A, *h_B, *h_C;
	h_A = (float *)malloc(M * K * sizeof(float));
	h_B = (float *)malloc(K * N * sizeof(float));
	h_C = (float *)malloc(M * N * sizeof(float));

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


	for (int i = 0; i < 5; i++) {
		// Allocate memory on the device
		float *d_A, *d_B, *d_C;
		start_time = clock();
		cudaMalloc(&d_A, M * K * sizeof(float));
		cudaMalloc(&d_B, K * N * sizeof(float));
		cudaMalloc(&d_C, M * N * sizeof(float));

		// Copy input matrices to the device
		cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_C, h_C, M * N * sizeof(float), cudaMemcpyHostToDevice);

		// Launch the kernel
		// create as many blocks as necessary to map all of C
		dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32), 1);
		// 32 * 32 = 1024 thread per block
		dim3 blockDim(32, 32, 1);
		// launch the asynchronous execution of the kernel on the device
		// The function call returns immediately on the host
		sgemm_naive<<<gridDim, blockDim>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
		cudaDeviceSynchronize();

		// Copy the result back to the host
		cudaMemcpy(h_A, d_A, M * K * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_B, d_B, K * N * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

		// Free memory on the device
		cudaFree(d_A);
		cudaFree(d_B);
		cudaFree(d_C);

		elapsed_time = clock() - start_time;
		total_time += elapsed_time;
	}

		printf("Averaged Elapsed Time : %.20f", total_time / (5 * CLOCKS_PER_SEC));

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