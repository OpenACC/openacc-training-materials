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
//#include "dot_acc.h"

void dot_acc(int *A, int *B, int *C, int m, int n)
{
	int temp;
#pragma acc parallel loop gang private(temp) \
	present(A, B, C)
	for( int i = 0; i < m; i++ ) {
		temp = 0;
#pragma acc loop vector reduction(+:temp)
		for( int j = 0; j < n; j++ ) {
			temp +=  A[i*n+j] * B[i*n+j];
		}
		C[i] = temp;
	}
}



