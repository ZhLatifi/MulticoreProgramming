#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define gpuErrchk(ans) { gpuAssert((ans), FILE, LINE); }
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

#define THREADS_PER_BLOCK 256

shared int shared_array[THREADS_PER_BLOCK];

global void prefix_sum_kernel(int *d_a, int n) {
int tid = threadIdx.x;
int block_size = blockDim.x;

// Local prefix sum within a block
for (int stride = 1; stride <= n / 2; stride *= 2) {
  if (tid < n - stride) {
    shared_array[tid] = d_a[tid] + d_a[tid + stride];
    __syncthreads();
    d_a[tid] = (tid == 0) ? 0 : shared_array[tid - stride];
  }
  __syncthreads();
}

// Final correction for the last element (if needed)
if (tid == 0 && n % 2 == 1) {
  d_a[n - 1] += d_a[n - 2];
}
}

int main(int argc, char *argv[]) {
double elapsed_time_sum = 0.0;

size_t n = 0;
printf("[-] Please enter N: ");
scanf("%lu", &n);

int *a = (int *)malloc(n * sizeof(int));

int *d_a;
gpuErrchk(cudaMalloc(&d_a, n * sizeof(int)));
gpuErrchk(cudaMemcpy(d_a, a, n * sizeof(int), cudaMemcpyHostToDevice));

int num_blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

for (int i = 0; i < EXP_NUM; i++) {
  fill_array(a, n);
  double starttime = clock();

  parallel_prefix_sum(d_a, n);

  double elapsedtime = (clock() - starttime) / (double)CLOCKS_PER_SEC;
  elapsed_time_sum += elapsedtime;
}


printf("average running time : %f\n", elapsed_time_sum / EXP_NUM);

gpuErrchk(cudaMemcpy(a, d_a, n * sizeof(int), cudaMemcpyDeviceToHost));
print_array(a, n);
gpuErrchk(cudaFree(d_a));
free(a);

return EXIT_SUCCESS;
}

void parallel_prefix_sum(int *a, size_t n) {
  int blockSize = THREADS_PER_BLOCK;
  int numBlocks = (n + blockSize - 1) / blockSize;

  prefix_sum_kernel << <numBlocks, blockSize >> > (d_a, n);

  gpuErrchk(cudaGetLastError());
  gpuErrchk(cudaDeviceSynchronize());
}

void fill_array(int *a, size_t n) {
  // ... (same as before)
}

void print_array(int *a, size_t n) {
  // ... (same as before)
}