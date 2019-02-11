#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <math.h>

extern "C" __device__
float dist_cuda(float a, float b) {
	return sqrt(a*a + b*b);
}
