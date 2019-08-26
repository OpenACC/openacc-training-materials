/*
 * Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
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
