#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <unistd.h>

extern void dot_cuda(int*, int*, int*, int, int);

int main()
{

	int i, j, m, n;
	int *A, *B, *C, *D, *E;

	srand(0);

	m = 4098;
	n = 4098;

	A = (int*) malloc( m*n * sizeof(int));
	B = (int*) malloc( m*n * sizeof(int));
	C = (int*) malloc(  m  * sizeof(int));
	D = (int*) malloc(  m  * sizeof(int));
	E = (int*) malloc(  m  * sizeof(int));

	for( i = 0; i < m; i++ ) {
		for( j = 0; j < n; j++ ) {
			A[i*n+j] = rand() % 100 + 1;
			B[i*n+j] = rand() % 100 + 1;
		}
	}

	dot(A, B, C, m, n);

	dot_acc(A, B, D, m, n);

	dot_cuda(A, B, E, m, n);

	for( i = 0; i < m; i++ ) {
		if( C[i] != D[i] || C[i] != E[i]) {
			printf("Error at index %i\n", i);
			return 0;
		}
	}

	printf("Program finished sucessfully.\n");
	return 0;

}
