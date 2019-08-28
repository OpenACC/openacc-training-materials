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
#include <mpi.h>
#include <openacc.h>

#define MAX(X,Y) ((X>Y) ? X:Y)
#define MIN(X,Y) ((X<Y) ? X:Y)

void blur5_mpi(unsigned char *imgData, unsigned char *out, long w, long h, long ch)
{
  long step = w*ch;
  long x, y;
  const int filtersize = 5;
  double filter[5][5] = 
  {
     1,  1,  1,  1,  1,
     1,  2,  2,  2,  1,
     1,  2,  3,  2,  1,
     1,  2,  2,  2,  1,
     1,  1,  1,  1,  1
  };
  // The denominator for scale should be the sum
  // of non-zero elements in the filter.
  double scale = 1.0 / 35.0;

  int nranks;
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  long rows_per_rank = (h+(nranks-1))/nranks;
  int sendcounts[nranks];
  int senddispls[nranks];
  int recvcounts[nranks];
  int recvdispls[nranks];
  long lower, upper, copyLower, copyUpper, size, copySize;
  if(rank == 0) {
    for(int r = 0; r < nranks; r++) {
      lower = r*rows_per_rank;
      upper = lower + rows_per_rank;
      copyLower = MAX(lower-(filtersize/2), 0);
      copyUpper = MIN(upper+(filtersize/2), h);
      sendcounts[r] = (int) ((copyUpper-copyLower)*step);
      senddispls[r] = (int) (copyLower*step);
      recvcounts[r] = (int) ((upper-lower)*step);
      recvdispls[r] = (int) (lower*step);
    }
  }

  lower = rank*rows_per_rank;
  upper = MIN(lower + rows_per_rank, h);
  size = (upper-lower)*step;
  copyLower = MAX(lower-(filtersize/2), 0);
  copyUpper = MIN(upper+(filtersize/2), h);
  copySize = (copyUpper-copyLower)*step;

  int ndevices = acc_get_num_devices(acc_device_default);
  acc_set_device_num(rank%ndevices, acc_device_default);

  MPI_Scatterv(imgData, sendcounts, senddispls, MPI_UNSIGNED_CHAR,
   &imgData[copyLower*step], (int) copySize, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

#pragma acc parallel loop copyin(imgData[copyLower*step:copySize], filter) \
 copyout(out[lower*step:size])
  for(y = lower; y < upper; y++) {
#pragma acc loop vector
    for(x = 0; x < w; x++) {
      double blue = 0.0, green = 0.0, red = 0.0;
#pragma acc loop seq
      for(int fy = 0; fy < filtersize; fy++) {
        long iy = y - (filtersize/2) + fy;
#pragma acc loop seq
        for (int fx = 0; fx < filtersize; fx++) {
          long ix = x - (filtersize/2) + fx;
          if( (iy<0)  || (ix<0) || 
              (iy>=h) || (ix>=w) ) continue;
          blue  += filter[fy][fx] * (double)imgData[iy * step + ix * ch];
          green += filter[fy][fx] * (double)imgData[iy * step + ix * ch + 1];
          red   += filter[fy][fx] * (double)imgData[iy * step + ix * ch + 2];
        }
      }
      out[y * step + x * ch]      = 255 - (scale * blue);
      out[y * step + x * ch + 1 ] = 255 - (scale * green);
      out[y * step + x * ch + 2 ] = 255 - (scale * red);
    }
  }

  MPI_Gatherv(&out[lower*step], (int) size, MPI_UNSIGNED_CHAR,
   out, recvcounts, recvdispls, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
}

