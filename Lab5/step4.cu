#include <cuda_runtime.h>
#include <stdio.h>

// Define the gpuErrchk macro
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
	{
		if (code != cudaSuccess)
		{
			fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
			if (abort) exit(code);
		}
	}

__global__ void calcThreadKernel(int *block, int *warp, int *local_index) {
	int bd = blockDim.x;
	int bx = blockIdx.x;
	int tx = threadIdx.x;

	int global_idx = bd * bx + tx;
	block[global_idx] = bx;
	warp[global_idx] = tx / warpSize;
	local_index[global_idx] = tx;
}

int main(int argc, char **argv) {
	dim3 NUM_THREADS(64, 1, 1);
	dim3 NUM_BLOCKS(2, 1, 1);

	int size = NUM_THREADS.x * NUM_BLOCKS.x;
	int *block, *warp, *local_index;
	int *block_host, *warp_host, *local_index_host;

	gpuErrchk(cudaMalloc((void **)&block, size * sizeof(int)));
	gpuErrchk(cudaMalloc((void **)&warp, size * sizeof(int)));
	gpuErrchk(cudaMalloc((void **)&local_index, size * sizeof(int)));

	calcThreadKernel << <NUM_BLOCKS, NUM_THREADS >> > (block, warp, local_index);
	gpuErrchk(cudaGetLastError());

	gpuErrchk(cudaDeviceSynchronize());

	block_host = (int*)malloc(size * sizeof(int));
	warp_host = (int*)malloc(size * sizeof(int));
	local_index_host = (int*)malloc(size * sizeof(int));

	gpuErrchk(cudaMemcpy(block_host, block, size * sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(warp_host, warp, size * sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(local_index_host, local_index, size * sizeof(int), cudaMemcpyDeviceToHost));

	for (int i = 0; i < size; i++) {
		printf("Calculated Thread: %d,\tBlock: %d,\tWarp: %d,\tThread: %d\n", i, block_host[i], warp_host[i], local_index_host[i]);
	}

	gpuErrchk(cudaFree(block));
	gpuErrchk(cudaFree(warp));
	gpuErrchk(cudaFree(local_index));

	free(block_host);
	free(warp_host);
	free(local_index_host);

	return 0;
}