#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <unistd.h>

extern void dot_cuda(int*, int*, int*, int, int);

int main()
{

	int i, j, m, n;
	int *A, *B, *C, *D;

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

	dot(A, B, C, m, n);

#pragma acc data copyin(A[0:m*n],B[0:m*n]) copyout(D[0:m])
{
#pragma acc host_data use_device(A,B,D)
{
	dot_cuda(A, B, D, m, n);
}
}

	for( i = 0; i < m; i++ ) {
		if( C[i] != D[i]) {
			printf("Error at index %i\n", i);
			return 0;
		}
	}

	printf("Program finished sucessfully.\n");
	return 0;

}
