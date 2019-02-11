#include <stdio.h>
#define MAX(X,Y) ((X>Y) ? X:Y)
#define MIN(X,Y) ((X<Y) ? X:Y)
void blur5_serial(unsigned restrict char *imgData, unsigned restrict char *out, long w, long h, long ch)
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
  for(y = 0; y < h; y++) {
    for(x = 0; x < w; x++) {
      double blue = 0.0, green = 0.0, red = 0.0;
      for(int fy = 0; fy < filtersize; fy++) {
        long iy = y - (filtersize/2) + fy;
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
}

void blur5_parallel(unsigned restrict char *imgData, unsigned restrict char *out, long w, long h, long ch)
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
#pragma acc parallel loop copyin(imgData[:h*step]) copyout(out[:h*step])
  for(y = 0; y < h; y++) {
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
}

void blur5_blocked(unsigned restrict char *imgData, unsigned restrict char *out, long w, long h, long ch)
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

#pragma acc enter data create(out[:w*h*ch])
#pragma acc enter data copyin(imgData[:w*h*ch])

  const long numBlocks = 32;
  for(long block = 0; block < numBlocks; block++) {
    long lower = block*rowsPerBlock;
    long upper = MIN(h, lower+rowsPerBlock);
#pragma acc parallel loop present(imgData, out, filter)
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
  }

#pragma acc exit data delete(imgData) copyout(out[:w*h*ch])

}

void blur5_blocked_with_data(unsigned restrict char *imgData, unsigned restrict char *out, long w, long h, long ch)
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
#pragma acc update device(imgData[copyLower*step:(copyUpper-copyLower)*step])
      lastRowCopied = copyUpper;
    }
#pragma acc parallel loop present(imgData, out, filter)
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
#pragma acc update self(out[lower*step:(upper-lower)*step])
  }
#pragma acc exit data delete(imgData, out)
}

