#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <omp.h>
#include <mpi.h>

#include "imageWrapper.h"

extern "C" {
  void blur5(unsigned char*, unsigned char*, long, long, long);
  void blur5_serial(unsigned char*, unsigned char*, long, long, long);
  void blur5_parallel(unsigned char*, unsigned char*, long, long, long);
  void blur5_mpi(unsigned char*, unsigned char*, long, long, long);
}

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (argc < 3) {
    if(rank == 0) fprintf(stderr,"Usage: %s inFilename outFilename\n",argv[0]);
    return -1;
  }

  long w, h, ch;
  unsigned char* data;
  long dim[3];
  if(rank == 0) {
    data = readImage(argv[1], w, h, ch);
    dim[0] = w;
    dim[1] = h;
    dim[2] = ch;
  }
  MPI_Bcast(dim, 3, MPI_LONG, 0, MPI_COMM_WORLD);
  if(rank != 0) {
    w = dim[0];
    h = dim[1];
    ch = dim[2];
    data = (unsigned char*) malloc(w*h*ch*sizeof(unsigned char));
  }

  unsigned char* output1 = new unsigned char[w*h*ch];
  unsigned char* output2 = new unsigned char[w*h*ch];
  unsigned char* output3 = new unsigned char[w*h*ch];

  // Remove any overhead
  blur5_mpi(data, output3, w, h, ch);

  double st;
  if(rank == 0) st = omp_get_wtime();
  blur5(data, output1, w, h, ch);

  if(rank == 0) {
    printf("Time taken for blur5: %.4f seconds\n", omp_get_wtime()-st);
    printf("Running serial and baseline parallel for comparison...\n");
    st = omp_get_wtime();
    blur5_serial(data, output2, w, h, ch);
    printf("Time taken for serial blur5: %.4f seconds\n", omp_get_wtime()-st);

    st = omp_get_wtime();
    blur5_parallel(data, output3, w, h, ch);
    printf("Time taken for baseline parallel blur5: %.4f seconds\n", omp_get_wtime()-st);

    printf("Checking results for comparison...\n");
    int count = 0;
    bool success = true;
    for(int i = 0; i < w*h*ch; i++) {
      if(output1[i] != output2[i]) {
        success = false;
        count++;
        if(count > 10) break;
      }
    }

    if(success) {
      printf("Results are equal.\n");
    }
  }

  if(rank == 0) {
    memcpy(data, output1, w*h*ch*sizeof(unsigned char));
    writeImage(argv[2]);
  } else {
    delete[] data;
  }

  delete[] output1;
  delete[] output2;
  delete[] output3;

  MPI_Finalize();

  return 0;
}
