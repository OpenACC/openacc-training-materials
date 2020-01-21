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
#include <openacc.h>
#define MAX(X,Y) ((X>Y) ? X:Y)
#define MIN(X,Y) ((X<Y) ? X:Y)

void blur5_multi_device(unsigned char *imgData, unsigned char *out, long w, long h, long ch)
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

  int ndevices = acc_get_num_devices(acc_device_default);
  long rows_per_device = (h+(ndevices-1))/ndevices;

  long lower;
  long upper;
  long copyLower;
  long copyUpper;

  for(int device = 0; device < ndevices; device++) {
    acc_set_device_num(device, acc_device_default);
    lower = device*rows_per_device;
    upper = MIN(lower + rows_per_device, h);
    copyLower = MAX(lower-(filtersize/2), 0);
    copyUpper = MIN(upper+(filtersize/2), h);
#pragma acc enter data \
 create(imgData[copyLower*step:(copyUpper-copyLower)*step], \
        out[lower*step:(upper-lower)*step]) \
 copyin(filter[:5][:5])
  }

  for(int device = 0; device < ndevices; device++) {
    acc_set_device_num(device, acc_device_default);

    lower = device*rows_per_device;
    upper = MIN(lower + rows_per_device, h);
    copyLower = MAX(lower-(filtersize/2), 0);
    copyUpper = MIN(upper+(filtersize/2), h);

#pragma acc update device(imgData[copyLower*step:(copyUpper-copyLower)*step]) async
#pragma acc parallel loop present(filter, \
 imgData[copyLower*step:(copyUpper-copyLower)*step], \
 out[lower*step:(upper-lower)*step]) async
    for(y = lower; y < upper; y++) {
#pragma acc loop
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
#pragma acc update self(out[lower*step:(upper-lower)*step]) async
  }

  for(int device = 0; device < ndevices; device++) {
    acc_set_device_num(device, acc_device_default);
#pragma acc wait
    lower = device*rows_per_device;
    upper = MIN(lower + rows_per_device, h);
    copyLower = MAX(lower-(filtersize/2), 0);
    copyUpper = MIN(upper+(filtersize/2), h);
#pragma acc exit data delete(out[lower*step:(upper-lower)*step], \
 imgData[copyLower*step:(copyUpper-copyLower)*step], filter)
  }
}


