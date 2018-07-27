#include <stdio.h>
#include <stdlib.h>

void dot(int *A, int *B, int *C, int m, int n) 
{
	for( int i = 0; i < m; i++ ) {
		C[i] = 0;
		for( int j = 0; j < n; j++ ) {
			C[i] +=  A[i*n+j] * B[i*n+j];
		}
	}
}




