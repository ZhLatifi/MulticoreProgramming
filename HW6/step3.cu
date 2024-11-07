#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "cuda.h"
#include <time.h>

#define BLOCK_SIZE 32
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))
double start_time, elapsed_time;
double total_time = 0;
const int BM = 128; // Block size (rows)
const int BN = 128; // Block size (columns)
const int BK = 8; // Shared memory cache size for matrix A
const int TM = 8;  // Rows processed by each thread
const int TN = 8;  // Columns processed by each thread

__global__ void __launch_bounds__((BM * BN) / (TM * TN), 1)
    sgemm2DBlocktiling(int M, int N, int K, float alpha, const float* A,
					   const float* B, float beta, float* C) {
  const int cRow = blockIdx.y;
  const int cCol = blockIdx.x;

  const int totalResultsBlocktile = BM * BN;
  // A thread is responsible for calculating TM*TN elements in the blocktile
  const int numThreadsBlocktile = totalResultsBlocktile / (TM * TN);

  // ResultsPerBlock / ResultsPerThread == ThreadsPerBlock
  //assert(numThreadsBlocktile == blockDim.x);

  // BN/TN are the number of threads to span a column
  const int threadCol = threadIdx.x % (BN / TN);
  const int threadRow = threadIdx.x / (BN / TN);

  // allocate space for the current blocktile in smem
  extern __shared__ float As[BM * BK];
  extern __shared__ float Bs[BK * BN];

  // Move blocktile to beginning of A's row and B's column
  A += cRow * BM * K;
  B += cCol * BN;
  C += cRow * BM * N + cCol * BN;

  // calculating the indices that this thread will load into SMEM
  const int innerRowA = threadIdx.x / BK;
  const int innerColA = threadIdx.x % BK;
  // calculates the number of rows of As that are being loaded in a single step
  // by a single block
  const int strideA = numThreadsBlocktile / BK;
  const int innerRowB = threadIdx.x / BN;
  const int innerColB = threadIdx.x % BN;
  // for both As and Bs we want each load to span the full column-width, for
  // better GMEM coalescing (as opposed to spanning full row-width and iterating
  // across columns)
  const int strideB = numThreadsBlocktile / BN;

  // allocate thread-local cache for results in registerfile
  float threadResults[TM * TN] = {0.0};
  // register caches for As and Bs
  float regM[TM] = {0.0};
  float regN[TN] = {0.0};

  // outer-most loop over block tiles
  for (int bkIdx = 0; bkIdx < K; bkIdx += BK) {
    // populate the SMEM caches
    for (int loadOffset = 0; loadOffset < BM; loadOffset += strideA) {
      As[(innerRowA + loadOffset) * BK + innerColA] =
          A[(innerRowA + loadOffset) * K + innerColA];
    }
    for (int loadOffset = 0; loadOffset < BK; loadOffset += strideB) {
      Bs[(innerRowB + loadOffset) * BN + innerColB] =
          B[(innerRowB + loadOffset) * N + innerColB];
    }
    __syncthreads();

    // advance blocktile
    A += BK;     // move BK columns to right
    B += BK * N; // move BK rows down

    // calculate per-thread results
    for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
      // block into registers
      for (int i = 0; i < TM; ++i) {
        regM[i] = As[(threadRow * TM + i) * BK + dotIdx];
      }
      for (int i = 0; i < TN; ++i) {
        regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
      }
      for (int resIdxM = 0; resIdxM < TM; ++resIdxM) {
        for (int resIdxN = 0; resIdxN < TN; ++resIdxN) {
          threadResults[resIdxM * TN + resIdxN] +=
              regM[resIdxM] * regN[resIdxN];
        }
      }
    }
    __syncthreads();
  }

  // write out the results
  for (int resIdxM = 0; resIdxM < TM; ++resIdxM) {
    for (int resIdxN = 0; resIdxN < TN; ++resIdxN) {
      C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN] =
          alpha * threadResults[resIdxM * TN + resIdxN] +
          beta * C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN];
    }
  }
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
		size_t sharedMemSize = 2 * BLOCK_SIZE * BLOCK_SIZE * sizeof(float);

		start_time = clock();
		sgemm2DBlocktiling <<<gridDim, blockDim, sharedMemSize >>> (M, N, K, alpha, d_A, d_B, beta, d_C);
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