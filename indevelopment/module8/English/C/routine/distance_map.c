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
