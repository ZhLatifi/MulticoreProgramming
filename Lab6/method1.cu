#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
//#include <cuda.h>
#include <device_functions.h>
#include <cuda_device_runtime_api.h>
#include <device_launch_parameters.h>
#include <time.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

void fill_array(int *a, size_t n);
void print_array(int *a, size_t n);
void parallel_prefix_sum(int *a, size_t n);

#define EXP_NUM 1


__global__ void prefix_sum_kernel(int *d_a, int *d_last_sums, int n, int num_threads, int numBlocks) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= num_threads) return;

	int chunk_size = n / (num_threads * numBlocks);
	int start = tid * chunk_size;
	int end = (tid == num_threads - 1) ? n : start + chunk_size;

	// Local prefix sum over subarray
	for (int i = start + 1; i < end; i++) {
		d_a[i] += d_a[i - 1];
	}

	__syncthreads();

	if (tid == 0) {
		d_last_sums[0] = 0;
		for (int i = 1; i < num_threads; i++) {
			d_last_sums[i] = d_a[(i * chunk_size) - 1] + d_last_sums[i - 1];
		}
	}

	__syncthreads();

	if (tid != 0) {
		for (int i = start; i < end; i++) {
			d_a[i] += d_last_sums[tid];
		}
	}
}







int main(int argc, char *argv[]) {
	double elapsed_time_sum = 0.0;

	size_t n = 0;
	printf("[-] Please enter N: ");
	scanf("%lu", &n);

	int *a = (int *)malloc(n * sizeof(int));

	int *d_a, *d_last_sums;
	gpuErrchk(cudaMalloc(&d_a, n * sizeof(int)));
	gpuErrchk(cudaMemcpy(d_a, a, n * sizeof(int), cudaMemcpyHostToDevice));

	int num_threads = 8;
	gpuErrchk(cudaMalloc(&d_last_sums, num_threads * sizeof(int)));

	for (int i = 0; i < EXP_NUM; i++) {
		fill_array(a, n);
		double starttime = clock();

		parallel_prefix_sum(a, n);

		double elapsedtime = (clock() - starttime) / (double)CLOCKS_PER_SEC;
		elapsed_time_sum += elapsedtime;
	}


	printf("average running time : %f\n", elapsed_time_sum / EXP_NUM);

	//gpuErrchk(cudaFree(d_a));
	//gpuErrchk(cudaFree(d_last_sums));
	free(a);

	return EXIT_SUCCESS;
}

void parallel_prefix_sum(int *a, size_t n) {
	int *d_a, *d_last_sums;



	int num_threads = 8;
	gpuErrchk(cudaMalloc(&d_a, n * sizeof(int)));
	gpuErrchk(cudaMemcpy(d_a, a, n * sizeof(int), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMalloc(&d_last_sums, num_threads * sizeof(int)));

	int blockSize = num_threads;
	int numBlocks = 1;

	prefix_sum_kernel << <numBlocks, blockSize >> > (d_a, d_last_sums, n, num_threads, numBlocks);
	//prefix_sum_kernel << <numBlocks, blockSize,blockSize*sizeof(int) >> > (d_a, d_last_sums, n, num_threads, numBlocks);

	gpuErrchk(cudaGetLastError());

	gpuErrchk(cudaDeviceSynchronize());


	gpuErrchk(cudaMemcpy(a, d_a, n * sizeof(int), cudaMemcpyDeviceToHost));
	//print_array(a, n);
	gpuErrchk(cudaFree(d_a));
	gpuErrchk(cudaFree(d_last_sums));
}

void fill_array(int *a, size_t n) {
	for (size_t i = 0; i < n; ++i) {
		a[i] = i + 1;
	}
}

void print_array(int *a, size_t n) {
	printf("[-] array: ");
	for (size_t i = 0; i < n; ++i) {
		printf("%d, ", a[i]);
	}
	printf("\b\b  \n");
}





