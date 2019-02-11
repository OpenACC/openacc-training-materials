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
	int *A_d, *B_d, *C_d, size;
	cudaMalloc((void **)&A_d, m*n*sizeof(int));
	cudaMalloc((void **)&B_d, m*n*sizeof(int));
	cudaMalloc((void **)&C_d, m*sizeof(int));

	cudaMemcpy(A_d, A, m*n*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(B_d, B, m*n*sizeof(int), cudaMemcpyHostToDevice);

	size = min(512,n);

	dot_kernel<<<m,size,size*sizeof(int)>>>(A_d,B_d,C_d,m,n);
	cudaThreadSynchronize();

	cudaMemcpy(C, C_d, m*sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(A_d); cudaFree(B_d); cudaFree(C_d);

}
