#include <stdio.h>
#include <stdlib.h>
#include <math.h>

extern float dist_cuda(float, float);

#pragma acc routine seq
float dist_acc(float a, float b) {
	return sqrt(a*a + b*b);
}

void distance_map_acc(float *A, float *B, float *C, int m, int n)
{
#pragma acc parallel loop copyin(A[0:m],B[0:n]) \
copyout(C[0:m*n])
	for( int i = 0; i < m; i++ ) {
#pragma acc loop
		for( int j = 0; j < n; j++ ) {
			C[i*n+j] = dist_cuda(A[i], B[j]);
		}
	}
	
}
