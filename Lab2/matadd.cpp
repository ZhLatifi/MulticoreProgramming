/*
*	In His Exalted Name
*	Matrix Addition - Sequential Code
*	Ahmad Siavashi, Email: siavashi@aut.ac.ir
*	15/04/2018
*/

// Let it be.
#define _CRT_SECURE_NO_WARNINGS

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <omp.h>

typedef struct {
	int* A, *B, *C;
	int n, m;
} DataSet;



void fillDataSet(DataSet *dataSet);
void printDataSet(DataSet dataSet);
void closeDataSet(DataSet dataSet);
void addSerial(DataSet dataSet);
void add1D(DataSet dataSet);
void add2D(DataSet dataSet);
void block_add(DataSet dataSet, int i, int j);

#define BLOCK_SIZE 32
#define EXP_NUM 10

int main(int argc, char *argv[]) {


	#ifndef _OPENMP
		printf("OpenMP is not supported, sorry!\n");
		getchar();
		return 0;
	#endif 
	double starttime, elapsedtime;
	double elapsed_time_sum = 0.0;

	DataSet dataSet;
	if (argc < 3) {
		printf("[-] Invalid No. of arguments.\n");
		printf("[-] Try -> <n> <m> \n");
		printf(">>> ");
		scanf("%d %d", &dataSet.n, &dataSet.m);
	}
	else {
		dataSet.n = atoi(argv[1]);
		dataSet.m = atoi(argv[2]);
	}

	fillDataSet(&dataSet);
	for (int i = 0; i < EXP_NUM; i++) {
		double starttime = omp_get_wtime();
		//addSerial(dataSet);
		//add1D(dataSet);
		add2D(dataSet);
		double elapsedtime = omp_get_wtime() - starttime;

		elapsed_time_sum += elapsedtime;
		//printDataSet(dataSet);
	}
	printDataSet(dataSet);
	printf("average running time : %f\n", elapsed_time_sum / EXP_NUM);
	
	closeDataSet(dataSet);
	//system("PAUSE");
	return EXIT_SUCCESS;
}

void fillDataSet(DataSet *dataSet) {
	int i, j;

	dataSet->A = (int *)malloc(sizeof(int) * dataSet->n * dataSet->m);
	dataSet->B = (int *)malloc(sizeof(int) * dataSet->n * dataSet->m);
	dataSet->C = (int *)malloc(sizeof(int) * dataSet->n * dataSet->m);

	srand(time(NULL));

	for (i = 0; i < dataSet->n; i++) {
		for (j = 0; j < dataSet->m; j++) {
			dataSet->A[i*dataSet->m + j] = rand() % 100;
			dataSet->B[i*dataSet->m + j] = rand() % 100;
		}
	}
}

void printDataSet(DataSet dataSet) {
	int i, j;

	printf("[-] Matrix A\n");
	for (i = 0; i < dataSet.n; i++) {
		for (j = 0; j < dataSet.m; j++) {
			printf("%-4d", dataSet.A[i*dataSet.m + j]);
		}
		putchar('\n');
	}

	printf("[-] Matrix B\n");
	for (i = 0; i < dataSet.n; i++) {
		for (j = 0; j < dataSet.m; j++) {
			printf("%-4d", dataSet.B[i*dataSet.m + j]);
		}
		putchar('\n');
	}

	printf("[-] Matrix C\n");
	for (i = 0; i < dataSet.n; i++) {
		for (j = 0; j < dataSet.m; j++) {
			printf("%-8d", dataSet.C[i*dataSet.m + j]);
		}
		putchar('\n');
	}
}

// Serial
void closeDataSet(DataSet dataSet) {
	free(dataSet.A);
	free(dataSet.B);
	free(dataSet.C);
}


void addSerial(DataSet dataSet) {
	int i, j;
	for (i = 0; i < dataSet.n; i++) {
		for (j = 0; j < dataSet.m; j++) {
			dataSet.C[i * dataSet.m + j] = dataSet.A[i * dataSet.m + j] + dataSet.B[i * dataSet.m + j];
		}
	}
}

// 1D
void add1D(DataSet dataSet) {
	int i, j;
	#pragma omp parallel for private(j) num_threads(8)
	for (i = 0; i < dataSet.n; i++) {
		for (j = 0; j < dataSet.m; j++) {
			dataSet.C[i * dataSet.m + j] = dataSet.A[i * dataSet.m + j] + dataSet.B[i * dataSet.m + j];
		}
	}

}

// 2D
void block_add(DataSet dataSet, int i, int j) {
	for (int k = i * BLOCK_SIZE; k < (i + 1) * BLOCK_SIZE && k < dataSet.n; k++) {
		for (int p = j * BLOCK_SIZE; p < (j + 1) * BLOCK_SIZE && p < dataSet.m; p++) {
			dataSet.C[k * dataSet.m + p] = dataSet.A[k * dataSet.m + p] + dataSet.B[k * dataSet.m + p];
		}
	}
}
void add2D(DataSet dataSet) {
	int i, j;
	int row_block = (int)ceil(dataSet.n / (double)BLOCK_SIZE);
	int col_block = (int)ceil(dataSet.m / (double)BLOCK_SIZE);
	#pragma omp parallel for private (j) num_threads(8)
	for (i = 0; i < row_block; i++) {
		for (j = 0; j < col_block; j++) {
			block_add(dataSet, i, j);
		}
	}
}