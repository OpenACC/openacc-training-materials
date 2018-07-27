#include <stdio.h>
#include <stdlib.h>
#include <openacc.h>


void map(int* A, int* A_d, int size) {
	acc_map_data(A, A_d, size);
}

void unmap(int* A) { // Host pointer
	acc_unmap_data(A);
}

