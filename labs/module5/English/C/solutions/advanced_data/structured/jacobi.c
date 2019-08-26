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

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "laplace2d.h"

int main(int argc, char** argv)
{

    int n, m, iter_max;

    if(argc > 1){
        n = atoi(argv[1]);
    } else {
        n = 4096;
    }

    if(argc > 2){
        m = atoi(argv[2]);
    } else {
        m = 4096;
    }

    if(argc > 3){
        iter_max = atoi(argv[3]);
    } else {
        iter_max = 1000;
    }

    const double tol = 1.0e-6;
    double error = 1.0;

    double *restrict A    = (double*)malloc(sizeof(double)*n*m);
    double *restrict Anew = (double*)malloc(sizeof(double)*n*m);
    
    initialize(A, Anew, m, n);
        
    printf("Jacobi relaxation Calculation: %d x %d mesh\n", n, m);
    
    double st = omp_get_wtime();
    int iter = 0;

    #pragma acc data copyin(A[:m*n]) create(Anew[:m*n])
    {
        while ( error > tol && iter < iter_max )
        {
            error = calcNext(A, Anew, m, n);
            swap(A, Anew, m, n);

            if(iter % 100 == 0) printf("%5d, %0.6f\n", iter, error);
        
            iter++;

        }
    }

    double runtime = omp_get_wtime() - st;
 
    printf(" total: %f s\n", runtime);

    deallocate(A, Anew);

    return 0;
}
