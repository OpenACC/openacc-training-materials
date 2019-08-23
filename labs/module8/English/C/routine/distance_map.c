#include <stdio.h>
#include <stdlib.h>
#include <math.h>

float dist(float a, float b) {
	return sqrt(a*a + b*b);
}

void distance_map(float *A, float *B, float *C, int m, int n)
{

	for( int i = 0; i < m; i++ ) {
		for( int j = 0; j < n; j++ ) {
			C[i*n+j] = dist(A[i], B[j]);
		}
	}

}
