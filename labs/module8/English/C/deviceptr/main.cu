#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

extern "C" void dot_acc(int*, int*, int*, int, int);
extern "C" void dot(int*, int*, int*, int, int);

int main()
{

	int i, j, m, n;
	int *A, *B, *C, *D;
	int *A_d, *B_d, *C_d;

	srand(0);

	m = 4098;
	n = 4098;

	A = (int*) malloc( m*n * sizeof(int));
	B = (int*) malloc( m*n * sizeof(int));
	C = (int*) malloc(  m  * sizeof(int));
	D = (int*) malloc(  m  * sizeof(int));

	for( i = 0; i < m; i++ ) {
		for( j = 0; j < n; j++ ) {
			A[i*n+j] = rand() % 100 + 1;
			B[i*n+j] = rand() % 100 + 1;
		}
	}

	cudaMalloc((void **)&A_d, m*n*sizeof(int));
	cudaMalloc((void **)&B_d, m*n*sizeof(int));
	cudaMalloc((void **)&C_d, m*  sizeof(int));

	cudaMemcpy(A_d, A, m*n*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(B_d, B, m*n*sizeof(int), cudaMemcpyHostToDevice);

	dot_acc(A_d,B_d,C_d,m,n);

	cudaMemcpy(C, C_d, m*sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(A_d); cudaFree(B_d); cudaFree(C_d);

	dot(A,B,D,m,n);

	for( i = 0; i < m; i++ ) {
		if( C[i] != D[i] ) {
			printf("Error at index %i\n", i);
			return 0;
		}
	}

	printf("Program finished sucessfully.\n");
	return 0;

}
