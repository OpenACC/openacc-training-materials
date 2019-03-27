#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <omp.h>

#include "imageWrapper.h"

extern "C" {
  void filter_f90_blur5_(unsigned char*, unsigned char*, long, long, long);
  void filter_f90_blur5_serial_(unsigned char*, unsigned char*, long, long, long);
  void filter_f90_blur5_parallel_(unsigned char*, unsigned char*, long, long, long);
  void filter_pipeline_f90_blur5_pipelined_(unsigned char*, unsigned char*, long, long, long);
}

#define blur5 filter_f90_blur5_
#define blur5_serial filter_f90_blur5_serial_
#define blur5_parallel filter_f90_blur5_parallel_
#define blur5_pipelined filter_pipeline_f90_blur5_pipelined_

int main(int argc, char** argv)
{
  if (argc < 3) {
    fprintf(stderr,"Usage: %s inFilename outFilename\n",argv[0]);
    return -1;
  }

  long w, h, ch;
  unsigned char* data = readImage(argv[1], w, h, ch);

  unsigned char* output1 = new unsigned char[w*h*ch];
  unsigned char* output2 = new unsigned char[w*h*ch];
  unsigned char* output3 = new unsigned char[w*h*ch];

  // Warm Up
  double st = omp_get_wtime();
  blur5_serial(data, output1, w, h, ch);
  printf("Time taken for serial blur5: %.4f seconds\n", omp_get_wtime()-st);
  blur5_pipelined(data, output3, w, h, ch);

  st = omp_get_wtime();
  blur5(data, output2, w, h, ch);
  printf("Time taken for parallel blur5: %.4f seconds\n", omp_get_wtime()-st);

  st = omp_get_wtime();
  blur5_parallel(data, output3, w, h, ch);
  printf("Time taken for baseline parallel blur5: %.4f seconds\n", omp_get_wtime()-st);

  printf("Checking results for comparison...\n");
  bool success = true;
  int counter = 0;
  for(int i = 0; i < w*h*ch; i++) {
    if(output1[i] != output2[i]) {
      printf("Error at index %d: See %u, expected %u\n", i, output1[i], output2[i]);
      success = false;
      counter++;
      if(counter > 10) break;
    }
  }

  if(success) {
    printf("Code results are correct.\n");
  }

  memcpy(data, output2, w*h*ch*sizeof(unsigned char));
  writeImage(argv[2]);

  delete[] output1;
  delete[] output2;
  delete[] output3;

  return 0;
}
