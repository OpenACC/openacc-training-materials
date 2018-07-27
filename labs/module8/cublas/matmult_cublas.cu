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
