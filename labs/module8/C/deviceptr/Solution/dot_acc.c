#include <stdio.h>
#include <stdlib.h>

void dot_acc(int *A, int *B, int *C, int m, int n)
{
	int temp;
#pragma acc parallel loop gang private(temp) \
	deviceptr(A,B,C)
	for( int i = 0; i < m; i++ ) {
		temp = 0;
#pragma acc loop vector reduction(+:temp)
		for( int j = 0; j < n; j++ ) {
			temp +=  A[i*n+j] * B[i*n+j];
		}
		C[i] = temp;
	}
}



