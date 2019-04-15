#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>

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
			A[i*m+j] = (rand()/(float)RAND_MAX);
	for( i = 0; i < n; i++ ) // loop through all columns
		for( j = 0; j < k; j++ ) // loop through all rows
			B[i*k+j] = (rand()/(float)RAND_MAX);

	matmult(A, B, C, m, k, n);

#pragma acc data copyin(A[:m*k], B[:k*n]) copyout(D[:m*n])
{
	#pragma acc host_data use_device(A,B,D)
	{
		matmult_cublas(A, B, D, m, k, n);
	}
}

        int count = 0;
        int success = 1;
        for( i = 0; i < m*n; i++ ) {
                if( fabsf(C[i] - D[i]) > 0.001) {
                        printf("Error at index %d, %.3f vs. %.3f\n", i, C[i], D[i]);
                        count++;
                        success = 0;
                        if(count > 10) break;
                }
        }

        if(success)
                printf("Program finished sucessfully.\n");
        return 0;

}
