#include <stdio.h>
#include <openacc.h>
#include <omp.h>
#define MAX(X,Y) ((X>Y) ? X:Y)
#define MIN(X,Y) ((X<Y) ? X:Y)

void blur5_openmp(unsigned char *imgData, unsigned char *out, long w, long h, long ch)
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

#pragma omp parallel for num_threads(ndevices)
  for(int device = 0; device < ndevices; device++) {
    long lower = device*rows_per_device;
    long upper = MIN(lower + rows_per_device, h);
    long copyLower = MAX(lower-(filtersize/2), 0);
    long copyUpper = MIN(upper+(filtersize/2), h);

    int tid = omp_get_thread_num();
    acc_set_device_num(tid, acc_device_default);

#pragma acc parallel loop \
 copyin(imgData[copyLower*step:(copyUpper-copyLower)*step], filter) \
 copyout(out[lower*step:(upper-lower)*step])
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
  } // end omp parallel for
}

