
# Data Management with OpenACC

The following timer counts down to a five minute warning before the lab instance shuts down.  You should get a pop up at the five minute warning reminding you to save your work!  If you are about to run out of time, please see the [Post-Lab](#Post-Lab-Summary) section for saving this lab to view offline later.

---

## Introduction

Our goal for this lab is to learn what exactly code profiling is, and how we can use it to help us write powerful parallel programs.  
  
  
  
![development-cycle.png](../images/development-cycle.png)

This is the OpenACC 3-Step development cycle.

**Analyze** your code, and predict where potential parallelism can be uncovered. Use profiler to help understand what is happening in the code, and where parallelism may exist.

**Parallelize** your code, starting with the most time consuming parts. Focus on maintaining correct results from your program.

**Optimize** your code, focusing on maximizing performance. Performance may not increase all-at-once during early parallelization.

We are currently tackling the **analyze** step. We will use PGI's code profiler to get an understanding of a relatively simple sample code before moving onto the next two steps.

## Run the Code

Our first step to analyzing this code is to run it. We need to record the results of our program before making any changes so that we can compare them to the results from the parallel code later on. It is also important to record the time that the program takes to run, as this will be our primary indicator to whether or not our parallelization is improving performance.


```python
!pgcc -fast -o laplace jacobi.c laplace2d.c && echo "Compilation Successful!" && ./laplace
```

    jacobi.c:
    laplace2d.c:
    Compilation Successful!
    Jacobi relaxation Calculation: 4096 x 4096 mesh
        0, 0.250000
      100, 0.002397
      200, 0.001204
      300, 0.000804
      400, 0.000603
      500, 0.000483
      600, 0.000403
      700, 0.000345
      800, 0.000302
      900, 0.000269
     total: 49.446116 s
    

### Optional: Compiling Code

We are using the PGI compiler to compiler our code. You will not need to memorize the compiler commands to complete this lab, however, they will be helpful to know if you want to parallelize your own personal code with OpenACC.

**pgcc**      : this is the command to compile C code  
**pgc++**     : this is the command to compile C++ code  
**pgfortran** : this is the command to compile Fortran code  
**-fast**     : this compiler flag will allow the compiler to perform additional optimizations to our code

### Understanding Code Results

TODO

## Analyze the Code

Now that we know how long the code took to run, and what the code's output looks like, we should be able to view the code with a decent idea of what is happening. The code is contained within two files, which you may open and view.

[jacobi.c](../../view/C/jacobi.c)  
[laplace2d.c](../../view/C/laplace2d.c)  
  
You may read through these two files on your own, but we will also highlight the most important parts below in the "Code Breakdown".

### Optional: Code Theory

The code simulates heat distribution across a 2-dimensional metal plate. In the beginning, the plate will be unheated, meaning that the entire plate will be room temperature. Then, a constant heat will be applied to the edge of the plate, then the code will simulate that heat distributing across the plate.  

This is a visual representation of the plate before the simulation starts:  
  
![plate1.png](../images/plate1.png)  
  
We can see that the plate is uniformly room temperature, except for the top edge. Within the [laplace2d.c](../../view/C/laplace2d.c) file, we see a function called **initialize**. This function is what "heats" the top edge of the plate. 
  
```
void initialize(double *restrict A, double *restrict Anew, int m, int n)  
{  
    memset(A, 0, n * m * sizeof(double));  
    memset(Anew, 0, n * m * sizeof(double));  
  
    for(int i = 0; i < m; i++){  
        A[i] = 1.0;  
        Anew[i] = 1.0;  
    }  
}  
```

After the top edge is heated, the code will simulate that heat distributing across the length of the plate.  
This is the plate after several iterations of our simulation:  
  
![plate2.png](../images/plate2.png) 

That's the theory: simple heat distribution. However, we are more interested in how the code works. 

### Code Breakdown

The 2-dimensional plate is represented by a 2-dimensional array containing double values. These doubles represent temperature; 0.0 is room temperature, and 1.0 is our max temperature. The 2-dimensional plate has two states, one represents the current temperature, and one represents the simulated, updated temperature. These two states are represented by arrays **A** and **Anew** respectively. The following is a visual representation of these arrays, with the top edge "heated".

![plate_sim2.png](../images/plate_sim2.png)  
    
    
    
The distinction between these two arrays is very important for our **calcNext** function. Our calcNext is essentially our "simulate" function. calcNext will look at the inner elements of A (meaning everything except for the edges of the plate) and update each elements temperature based on the temperature of its neighbors.  

![plate_sim3.png](../images/plate_sim3.png)  

This is the **calcNext** function:
```
double calcNext(double *restrict A, double *restrict Anew, int m, int n)
{
    double error = 0.0;  
    for( int j = 1; j < n-1; j++)  
    {  
        for( int i = 1; i < m-1; i++ )   
        {  
            Anew[OFFSET(j, i, m)] = 0.25 * ( A[OFFSET(j, i+1, m)] + A[OFFSET(j, i-1, m)]  
                                           + A[OFFSET(j-1, i, m)] + A[OFFSET(j+1, i, m)]);  
            error = fmax( error, fabs(Anew[OFFSET(j, i, m)] - A[OFFSET(j, i , m)]));  
        }  
    }  
    return error;  
}  
```

Lastly, our **swap** function will copy the contents of **Anew** to **A**.

```
void swap(double *restrict A, double *restrict Anew, int m, int n)
{	
    for( int j = 1; j < n-1; j++)
    {
        for( int i = 1; i < m-1; i++ )
        {
            A[OFFSET(j, i, m)] = Anew[OFFSET(j, i, m)];    
        }
    }
}
```

## Profile the Code

Now we should have a good idea of what the code is doing. It is time to profile the code, to get a better understanding of how the code is running performance-wise. To profile our code, we will be using PGPROF, which is a visual profiler that comes with the PGI compiler. You can run PGPROF through noVNC by <a href="/vnc" target="_blank">clicking this link</a>.

We will start by profiling our laplace executable that we created earlier (when we ran our code). Select File > New Session, then where is says File: Enter Executable File [required], we will select our ***laplace*** executable.

TODO  
INSERT SCREENSHOT  
FIGURE OUT WHY THE noVNC IS LOOKING AT A COMPLETELY DIFFERENT FILE SYSTEM  

## Step 2 - Express Parallelism

Within each of the routines identified above, express the available parallelism
to the compiler using either the `acc kernels` or `acc parallel loop`
directive. As an example, here's the OpenACC code to add to the `matvec` routine.

```
void matvec(const matrix& A, const vector& x, const vector &y) {

  unsigned int num_rows=A.num_rows;
  unsigned int *restrict row_offsets=A.row_offsets;
  unsigned int *restrict cols=A.cols;
  double *restrict Acoefs=A.coefs;
  double *restrict xcoefs=x.coefs;
  double *restrict ycoefs=y.coefs;

#pragma acc kernels
  {
    for(int i=0;i<num_rows;i++) {
      double sum=0;
      int row_start=row_offsets[i];
      int row_end=row_offsets[i+1];
      for(int j=row_start;j<row_end;j++) {
        unsigned int Acol=cols[j];
        double Acoef=Acoefs[j];
        double xcoef=xcoefs[Acol];
        sum+=Acoef*xcoef;
      }
      ycoefs[i]=sum;
    }
  }
}
```

Add the necessary directives to each routine **one at a time** in order of importance. After adding the directive, recompile the code, check that the answers have remained the same, and note the performance difference from your
change.

```
$ make
pgc++ -fast -acc -ta=tesla:managed -Minfo=accel main.cpp -o cg

matvec(const matrix &, const vector &, const vector &):
      8, include "matrix_functions.h"
          15, Generating copyout(ycoefs[:num_rows])
              Generating
copyin(xcoefs[:],Acoefs[:],cols[:],row_offsets[:num_rows+1])
          16, Loop is parallelizable
              Accelerator kernel generated
              Generating Tesla code
              16, #pragma acc loop gang, vector(128) /* blockIdx.x threadIdx.x */
          20, Loop is parallelizable
```

The performance may slow down as you're working on this step. Be sure
to read the compiler feedback to understand how the compiler parallelizes the
code for you. If you are doing the C/C++ lab, it may be necessary to declare
some pointers as `restrict` in order for the compiler to parallelize them. You
will know if this is necessary if the compiler feedback lists a "complex loop
carried dependency."

### Step 3 - Re-Profile Application

Once you have added the OpenACC directives to your code, you should obtain a
new profile of the application. For this step, use the NVIDIA Visual Profiler
to obtain a GPU timeline and see how the the GPU computation and data movement
from CUDA Unified Memory interact. 

- If you are doing this lab via qwikLABs, connect to the Ubuntu remote desktop using the <a href="/vnc" target="_blank">browser-based VNC client</a>

Once Visual Profiler has started, create a new session by selecting *File -> New
Session*. Then select the executable that you built by pressing the *Browse*
button next to *File*, browse to `/home/ubuntu/c99` or `/home/ubuntu/f90`, 
select `cg`,  and then press *Next*. On the next screen ensure that
*Enable unified memory profiling* is checked and press *Finish*. The result
should look like the image below. Experiment with Visual Profiler to see what
information you can learn from it.

![Image of NVIDIA Visual Profiler after completing lab 2 with the kernels
directive](https://github.com/NVIDIA-OpenACC-Course/nvidia-openacc-course-sources/raw/master/labs/lab2/visual_profiler_lab2.png)


## Conclusion

After completing the above steps for each of the 3 important routines your application should show a speed-up over the unaccelerated version. You can verify this by removing the `-ta` flag from your compiler options. 

If you have code like what is in the `solution.kernels` or `solution.parallel` directories, you should see a roughly 14% speed-up over the CPU version.  If you were to use a GPU such as a K40 vs the K520 in this g2.2xlarge instance, you can get speeds closer to 8.4 seconds!  Here's a table showing the speeds on different CPUs and GPUs:

| Processor | Time |
| --------- | ---- |
| Haswell CPU  | 30.519176 | 
| K40 GPU      | 8.460459 | 
| g2.2xlarge CPU | 36.647187 |
| g2.2xlarge GPU | 32.084089 |

In the next lecture and lab we will replace CUDA Unified Memory with explicit memory management using OpenACC and then further optimize the loops using the OpenACC `loop` directive.

## Bonus Task

1. If you used the `kernels` directive to express the parallelism in the code,
try again with the `parallel loop` directive. Remember, you will need to take
responsibility of identifying any reductions in the code. If you used 
`parallel loop`, try using `kernels` instead and observe the differences both in
developer effort and performance.

