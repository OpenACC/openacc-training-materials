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

#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <cublas_v2.h>

extern "C" void matmult_cublas(float*, float*, float*, int, int, int);

void matmult_cublas(float *A, float *B, float *C, int m, int k, int n)
{
	int lda=m, ldb=k, ldc=m;
	const float alf = 1.0;
	const float bet = 0.0;

	cublasHandle_t handle;
	cublasCreate(&handle);

	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
		m, n, k, &alf, A, lda, B, ldb, &bet, C, ldc);

	cublasDestroy(handle);

	return;

}
