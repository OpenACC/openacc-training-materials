#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main()
{

	int i, j, m, n;
	float *A, *B, *C, *D;

	srand(0);

	m = 4098;
	n = 4098;

	A = (float*) malloc( m   * sizeof(float));
	B = (float*) malloc( n   * sizeof(float));
	C = (float*) malloc( m*n * sizeof(float));
	D = (float*) malloc( m*n * sizeof(float));

	for( i = 0; i < m; i++ ) 
		A[i] = (float) (rand() % 100 + 1);
	for( i = 0; i < n; i++ )
		B[i] = (float) (rand() % 100 + 1);

	distance_map(A,B,C,m,n);
	
	distance_map_acc(A,B,D,m,n);

	for( i = 0; i < m*n; i++ ) {
		if( abs(C[i]-D[i]) > 0.0001) {
			printf("Error at index %i\n", i);
			return 0;
		}
	}

	printf("Program finished sucessfully.\n");
	return 0;

}
