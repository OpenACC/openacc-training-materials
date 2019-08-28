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
#define MAX(X,Y) ((X>Y) ? X:Y)
#define MIN(X,Y) ((X<Y) ? X:Y)

void blur5_pipelined(unsigned char *imgData, unsigned char *out, long w, long h, long ch)
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
#pragma acc declare copyin(filter)
  // The denominator for scale should be the sum
  // of non-zero elements in the filter.
  double scale = 1.0 / 35.0;
#pragma acc enter data create(imgData[:w*h*ch], out[:w*h*ch])

  const long numBlocks = 32;
  const long rowsPerBlock = (h+(numBlocks-1))/numBlocks;
  long lastRowCopied = 0; // Have not copied any rows yet
  for(long block = 0; block < numBlocks; block++) {
    long lower = block*rowsPerBlock; // Compute Lower
    long upper = MIN(h, lower+rowsPerBlock); // Compute Upper
    long copyLower = lastRowCopied+1; // Data copy lower
    long copyUpper = MIN(upper+(filtersize/2), h); // Data copy upper
    if(copyLower < copyUpper) {
#pragma acc update device(imgData[copyLower*step:(copyUpper-copyLower)*step]) async(5)
      lastRowCopied = copyUpper;
    }
#pragma acc wait(5)
#pragma acc parallel loop present(imgData, out) async(block%2)
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
#pragma acc update self(out[lower*step:(upper-lower)*step]) async(block%2)
  }
#pragma acc wait
#pragma acc exit data delete(imgData, out)
}

