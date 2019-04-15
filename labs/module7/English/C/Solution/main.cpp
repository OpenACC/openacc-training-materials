#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <omp.h>

#include "imageWrapper.h"

extern "C" {
  void blur5_serial(unsigned char*, unsigned char*, long, long, long);
  void blur5_parallel(unsigned char*, unsigned char*, long, long, long);
  void blur5_blocked(unsigned char*, unsigned char*, long, long, long);
  void blur5_blocked_with_data(unsigned char*, unsigned char*, long, long, long);
  void blur5_pipelined(unsigned char*, unsigned char*, long, long, long);
}

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

  double st = omp_get_wtime();
  blur5_serial(data, output1, w, h, ch);
  printf("Time taken for serial blur5: %.4f seconds\n", omp_get_wtime()-st);

  st = omp_get_wtime();
  blur5_pipelined(data, output2, w, h, ch);
  printf("Time taken for pipelined blur5: %.4f seconds\n", omp_get_wtime()-st);

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

  return 0;
}
