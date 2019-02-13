#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

extern void matmult_cublas(float*, float*, float*, int, int, int);

int main()
{

	int i, j, m, n, k;
	float *A, *B, *C, *D;

	srand(0);

	m = 256;
	n = 256;
	k = 256;

	A = (float*) malloc( m*k * sizeof(float));
	B = (float*) malloc( k*n * sizeof(float));
	C = (float*) malloc( m*n * sizeof(float));
	D = (float*) malloc( m*n * sizeof(float));

	for( i = 0; i < k; i++ ) // loop through all columns
		for( j = 0; j < m; j++ ) // loop through all rows
			A[i*m+j] = (float) i+j;
	for( i = 0; i < n; i++ ) // loop through all columns
		for( j = 0; j < k; j++ ) // loop through all rows
			B[i*k+j] = (float) i+j;

	matmult(A, B, C, m, k, n);

	matmult_cublas(A, B, D, m, k, n);

	for( i = 0; i < m*n; i++ ) {
		if( C[i] != D[i]) {
			printf("Error at index %i\n", i);
			return 0;
		}
	}

	printf("Program finished sucessfully.\n");
	return 0;

}
