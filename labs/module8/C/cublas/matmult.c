#include <stdio.h>
#include <stdlib.h>

void matmult(float *A, float *B, float *C, int m, int k, int n) 
{
	for( int i = 0; i < n; i++ ) {
		for( int j = 0; j < m; j++ ) {
			C[i*m+j] = 0;
			for( int k0 = 0; k0 < k; k0++ )
				C[i*m+j] += A[k0*m+j] * B[i*k+k0];
		}
	}
}




