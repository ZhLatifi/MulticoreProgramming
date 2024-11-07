/*
*				In His Exalted Name
*	Title:	Prefix Sum Sequential Code
*	Author: Ahmad Siavashi, Email: siavashi@aut.ac.ir
*	Date:	29/04/2018
*/

#define _CRT_SECURE_NO_WARNINGS

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <omp.h>

void omp_check();
void fill_array(int *a, size_t n);
void print_array(int *a, size_t n);
void prefix_sum(int *a, size_t n);
void parallel_prefix_sum(int *a, size_t n);
void hillis_prefix_sum(int *a, size_t n);

#define EXP_NUM 5


int main(int argc, char *argv[]) {
	double starttime, elapsedtime;
	double elapsed_time_sum = 0.0;
	// Check for correct compilation settings
	omp_check();
	// Input N

	size_t n = 0;
	printf("[-] Please enter N: ");
	scanf("%uld\n", &n);


	// Allocate memory for array
	int * a = (int *)malloc(n * sizeof a);
	int * b = (int *)malloc(n * sizeof b);


	// Fill array with numbers 1..n
	fill_array(a, n);

	// Print array
	//print_array(a, n);


	// Compute prefix sum

	for (int i = 0; i < EXP_NUM; i++) {
		double starttime = omp_get_wtime();

		prefix_sum(a, n);
		//parallel_prefix_sum(a, n);
		//hillis_prefix_sum(a, n);

		double elapsedtime = omp_get_wtime() - starttime;
		elapsed_time_sum += elapsedtime;
	}
	printf("average running time : %f\n", elapsed_time_sum / EXP_NUM);

	// Print array
	//print_array(a, n);

	// Free allocated memory
	free(a);
	system("pause");
	return EXIT_SUCCESS;
}


// serial function
void prefix_sum(int *a, size_t n) {

	int i;
	for (i = 1; i < n; ++i)
	{
		a[i] = a[i] + a[i - 1];
	}
}


// method 1 function
void parallel_prefix_sum(int *a, size_t n) {

	int i , j;
	int *last_sums, *starts, *ends;

#pragma omp parallel num_threads(8)
	{
		int tid = omp_get_thread_num();
		int num_threads = omp_get_num_threads();
#pragma omp single
		{
			last_sums = (int *)malloc(num_threads * sizeof(int));
			starts = (int *)malloc(num_threads * sizeof(int));
			ends = (int *)malloc(num_threads * sizeof(int));
		}


		int chunk_size = n / num_threads;
		starts[tid] = tid * chunk_size;
		ends[tid] = starts[tid] + chunk_size;

		// Local prefix sum over subarray
		for (int i = starts[tid] + 1; i < ends[tid]; i++) {
			a[i] += a[i - 1];
		}
		#pragma omp barrier


		#pragma omp single
		{
			last_sums[0] = 0;
			for (int i = 1; i < num_threads; i++) {
				last_sums[i] = a[starts[i] - 1] + last_sums[i - 1];
			}
		}
		if (tid != 0) {
			for (int i = starts[tid]; i < ends[tid]; i++) {
				a[i] += last_sums[tid];
			}
		}

	}
}

// method 2 function
void hillis_prefix_sum(int *a, size_t n) {
	int stride, i;
	int *partial_sum = malloc(n * sizeof(int));

	for (stride = 1; stride < n; stride *= 2) {
#pragma omp parallel num_threads(2) private(i)
		{
			int stop = n - stride;

#pragma omp for
			for (i = 0; i < stop; i++) {
				partial_sum[i + stride] = a[i + stride] + a[i];
			}

#pragma omp for
			for (i = stride; i < n; i++) {
				a[i] = partial_sum[i];
			}
		}
	}
	free(partial_sum);
}



void print_array(int *a, size_t n) {
	int i;
	printf("[-] array: ");
	for (i = 0; i < n; ++i) {
		printf("%d, ", a[i]);
	}
	printf("\b\b  \n");
}

void fill_array(int *a, size_t n) {
	int i;
	for (i = 0; i < n; ++i) {
		a[i] = i + 1;

	}
}

void omp_check() {
	printf("------------ Info -------------\n");
#ifdef _DEBUG
	printf("[!] Configuration: Debug.\n");
#pragma message ("Change configuration to Release for a fast execution.")
#else
	printf("[-] Configuration: Release.\n");
#endif // _DEBUG
#ifdef _M_X64
	printf("[-] Platform: x64\n");
#elif _M_IX86 
	printf("[-] Platform: x86\n");
#pragma message ("Change platform to x64 for more memory.")
#endif // _M_IX86 
#ifdef _OPENMP
	printf("[-] OpenMP is off.\n");
	printf("[-] OpenMP version: %d\n", _OPENMP);
#else
	printf("[!] OpenMP is on.\n");
	printf("[#] Enable OpenMP.\n");
#endif // _OPENMP
	printf("[-] Maximum threads: %d\n", omp_get_max_threads());
	printf("[-] Nested Parallelism: %s\n", omp_get_nested() ? "On" : "Off");
#pragma message("Enable nested parallelism if you wish to have parallel region within parallel region.")
	printf("===============================\n");
}
