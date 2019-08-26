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
#include <cuda.h>

__global__
void dot_kernel(int *A, int *B, int *C, int m, int n)
{

	extern __shared__ int temp[];
	int i = blockIdx.x;
	int j = threadIdx.x;

	if ( (i < m) && (j < n) ) temp[j] = A[i*n+j] * B[i*n+j];
	__syncthreads();

	int k = j + blockDim.x;
	while ( (i < m) && (k < n) ) 
	{
		temp[j] += A[i*n+k] * B[i*n+k];
		k += blockDim.x;
	}
	__syncthreads();

	for (int stride = blockDim.x/2; stride >  0; stride /= 2) {
		if (j < stride)
			temp[j] += temp[j + stride];
		__syncthreads();
	}

	C[blockIdx.x] = temp[0];
}


extern "C" void dot_cuda(int *A, int *B, int *C, int m, int n)
{
	int size = min(512,n);
	dot_kernel<<<m,size,size*sizeof(int)>>>(A,B,C,m,n);
	cudaDeviceSynchronize();

}
