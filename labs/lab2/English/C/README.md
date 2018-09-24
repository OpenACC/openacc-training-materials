
# Data Management with OpenACC

This version of the lab is intended for C/C++ programmers. The Fortran version of this lab is available [here](../Fortran/README.md).

You will receive a warning five minutes before the lab instance shuts down. Remember to save your work! If you are about to run out of time, please see the [Post-Lab](#Post-Lab-Summary) section for saving this lab to view offline later.

---
Let's execute the cell below to display information about the GPUs running on the server. To do this, execute the cell block below by giving it focus (clicking on it with your mouse), and hitting Ctrl-Enter, or pressing the play button in the toolbar above.  If all goes well, you should see some output returned below the grey cell.


```sh
!pgaccelinfo
```

---

## Introduction

Our goal for this lab is to use the OpenACC Data Directives to properly manage our data.
  
  
  
![development_cycle.png](../images/development_cycle.png)

This is the OpenACC 3-Step development cycle.

**Analyze** your code, and predict where potential parallelism can be uncovered. Use profiler to help understand what is happening in the code, and where parallelism may exist.

**Parallelize** your code, starting with the most time consuming parts. Focus on maintaining correct results from your program.

**Optimize** your code, focusing on maximizing performance. Performance may not increase all-at-once during early parallelization.

We are currently tackling the **parallelize** step. We will include the OpenACC data directive to properly manage data within our parallelized code.

---

## Run the Code (With Managed Memory)

In the previous lab, we ran our code with CUDA Managed Memory, and achieved a considerable performance boost. However, managed memory is not compatible with all GPUs, and it performs worse than programmer defined, proper memory management. Run the following script, and note the time the program takes to run. We are expecting that our own implementation will run a little bit better.


```sh
!pgcc -fast -ta=tesla:cc30,managed -Minfo=accel -o laplace_managed jacobi.c laplace2d.c && ./laplace_managed
```

### Optional: Analyze the Code

If you would like a refresher on the code files that we are working on, you may view both of them using the two links below.

[jacobi.c](jacobi.c)  
[laplace2d.c](laplace2d.c)  

### Optional: Profile the Code

If you would like to profile the code, you may select <a href="/vnc" target="_blank">this link.</a> This will open a noVNC window, then you may use PGPROF to profile our managed memory laplace code. The laplace_managed executable will be found in the /notebooks/C directory.

---

## OpenACC Structured Data Directive

The OpenACC data directives allow the programmer to explicitly manage the data on the device (in our case, the GPU). Specifically, the structured data directive will mark a static region of our code as a **data region**.

```
< Initialize data on host (CPU) >

#pragma acc data < data clauses >
{

    < Code >

}
```

Device memory allocation happens at the beginning of the region, and device memory deallocation happens at the end of the region. Additionally, any data movement from the host to the device (CPU to GPU) happens at the beginning of the region, and any data movement from the device to the host (GPU to CPU) happens at the end of the region. Memory allocation/deallocation and data movement is defined by which clauses the programmer includes. This is a list of the most important data clauses that we can use:

**copy** : copy( A[0:N] ) : Allocates memory on device and copies data from host to device when entering region and copies data back to the host when exiting region  
**copyin** : copyin( A[0:N] ) : Allocates memory on device and copies data from host to device when entering region  
**copyout** : copyout( A[0:N] ) : Allocates memory on device and copies data to the host when exiting region  
**create** : create( A[0:N] ) : Allocates memory on device but does not copy  
**present** : present( A ) : Data is already present on device from another containing data region  

All of these data clauses (except for present) will allocate device memory at the beginning of the data region, and deallocate device memory at the end of the data region. And with the exception of create, they will also transfer some amount of data between the host and device.

You may also use them to operate on multiple arrays at once, by including those arrays as a comma separated list.

```
#pragma acc data copy( A[0:N], B[0:M], C[0:Q] )
```

You may also use more than one data clause at a time.

```
#pragma acc data create( A[0:N] ) copyin( B[0:M] ) copyout( C[0:Q] )
```

These clauses can also be used directly with a parallel or kernels directive, because every parallel and kernels directive is surrounded by an **implied data region**.

```
#pragma acc kernels create(A[0:N]) copyin(B[0:M]) present(C[0:Q])
{
    < Code that uses A, B, and C >
}
```

### Encompassing Multiple Compute Regions

A single data region can contain any number of parallel/kernels regions. Take the following example:

```
#pragma acc data copyin(A[0:N], B[0:N]) create(C[0:N])
{

    #pragma acc parallel loop
    for( int i = 0; i < N; i++ )
    {
        C[i] = A[i] + B[i];
    }
    
    #pragma acc parallel loop
    for( int i = 0; i < N; i++ )
    {
        A[i] = C[i] + B[i];
    }

}
```

You may also encompass function calls within the data region:

```
void copy(int *A, int *B, int N)
{
    #pragma acc parallel loop
    for( int i = 0; i < N; i++ )
    {
        A[i] = B[i];
    }
}

...

#pragma acc data copyout(A[0:N],B[0:N]) copyin(C[0:N])
{
    copy(A, C, N);
    
    copy(A, B, N);
}
```

### Array Shaping

The "array shape" defines a portion of an array. Take the following example:

```
int *A = (int*) malloc(N * sizeof(int));

#pragma acc data create( A[0:N] )
```

The array shape is defined as [0:N], this means that the GPU copy will start at index 0, and be of size N. Array shape is of the format **Array[starting_index:size]**. Let's look at an example where we only want a portion of the array.

```
int *A = (int*) malloc(N * sizeof(int));

#pragma acc data create( A[0:N/2] )
```

In this example, the GPU copy will start at index 0, but will only be half the size of the CPU copy.

The shape of multi-dimensional arrays can be defined as follows:

```
#pragma acc data create( A[0:N][0:M] )
```

If you do not include a starting index, then 0 is assumed. For example:

```
#pragma acc data create( A[0:N] )
```

is equivalent to

```
#pragma acc data create( A[:N] )
```

### Host or Device Memory?

Here are two loops:

```
int *A = (int*) malloc(N * sizeof(int));

for (int i = 0; i < N; i++ )
{
    A[i] = 0;
}

#pragma acc parallel loop
for( int i = 0; i < N; i++ )
{
    A[i] = 1;
}
```

The first loop is not contained within an OpenACC compute region (a compute region is marked by either the parallel or kernels directive). Thus, **A[i]** will access host (CPU) memory.

The second loop is preceeded by the parallel directive, meaning that it is contained within an OpenACC compute region. **A[i]** in the second loop will access device (GPU) memory.

### Adding the Structured Data Directive to our Code

Use the following links to edit our laplace code. Add a structured data directive to properly handle the arrays **A** and **Anew**. 

[jacobi.c](../../../edit/04-Data-Management-with-OpenACC/C/jacobi.c)   
[laplace2d.c](../../../edit/04-Data-Management-with-OpenACC/C/laplace2d.c)  

Then, run the following script to check you solution. You code should run just as good as (or slightly better) than our managed memory code.


```sh
!pgcc -fast -ta=tesla:cc30 -Minfo=accel -o laplace_structured jacobi.c laplace2d.c && ./laplace_structured
```

If you are feeling stuck, or would like to check your answer, you can view the correct answer with the following link.

[jacobi.c](../../../edit/04-Data-Management-with-OpenACC/C/solutions/advanced_data/structured/jacobi.c)

### Optional: Profile the Code

If you would like to profile the code, you may select <a href="/vnc" target="_blank">this link.</a> This will open a noVNC window. To create a new session, select File > New Session. In the "File" section, select "Browse". Locate our **laplace_structured** executable in the /notebooks/C directory. The select OK > Next > Finished. The code will take a few seconds to finish profiling.

Take a moment to explore the profiler, and when you're ready, let's zoom in on the very beginning of our profile.

![structured_pgprof1.PNG](../images/structured_pgprof1.PNG)

We can see that we have uninterupted computation, and all of our data movement happens at the beginning of the program. This is ideal, because we are avoiding data transers in the middle of our computation.

---

## OpenACC Unstructured Data Directives

There are two unstructured data directives:

**enter data**: Handles device memory allocation, and copies from the Host to the Device. The two clauses that you may use with **enter data** are **create** for device memory allocation, and **copyin** for allocation, and memory copy.

**exit data**: Handles device memory deallocation, and copies from the Device to the Host. The two clauses that you may use with **exit data** are **delete** for device memory deallocation, and **copyout** for deallocation, and memory copy.

The unstructured data directives do not mark a "data region", because you are able to have multiple **enter data** and **exit data** directives in your code. It is better to think of them purely as memory allocation and deallocation.

The largest advantage of using unstructured data directives is their ability to branch across multiple functions. You may allocate your data in one function, and deallocate it in another. We can look at a simple example of that:

```
int* allocate(int size)
{
    int *ptr = (int*) malloc(size * sizeof(int));
    #pragma acc enter data create(ptr[0:size])
    return ptr;
}

void deallocate(int *ptr)
{
    #pragma acc exit data delete(ptr)
    free(ptr);
}

int main()
{
    int *ptr = allocate(100);
    
    #pragma acc parallel loop
    for( int i = 0; i < 100; i++ )
    {
        ptr[i] = 0;
    }
    
    deallocate(ptr);
}
```

Just like in the above code sample, you must first allocate the CPU copy of the array **before** you can allocate the GPU copy. Also, you must deallocate the GPU of the array **before** you deallocate the CPU copy.

### Adding Unstructured Data Directives to our Code

We are going to edit our code to use unstructured data directives to handle memory management. First, run the following script to reset your code to how it was before adding the structured data directive.


```sh
!cp ./solutions/basic_data/jacobi.c ./jacobi.c && cp ./solutions/basic_data/laplace2d.c ./laplace2d.c && echo "Reset Finished"
```

Now edit the code to use unstructured data directives. To fully utilize the unstructured data directives, try to get the code working by only altering the **laplace2d.c** code.

[jacobi.c](../../../edit/04-Data-Management-with-OpenACC/C/jacobi.c)   
[laplace2d.c](../../../edit/04-Data-Management-with-OpenACC/C/laplace2d.c)  

Run the following script to check your solution. Your code should run as fast as our structured implementation.


```sh
!pgcc -fast -ta=tesla:cc30 -Minfo=accel -o laplace_unstructured jacobi.c laplace2d.c && ./laplace_unstructured
```

If you are feeling stuck, or would like to check your answer, you can view the correct answer with the following link.

[laplace2d.c](../../../edit/04-Data-Management-with-OpenACC/C/solutions/advanced_data/unstructured/laplace2d.c)

### Optional: Profile the Code

If you would like to profile the code, you may select <a href="/vnc" target="_blank">this link.</a> This will open a noVNC window. To create a new session, select File > New Session. In the "File" section, select "Browse". Locate our **laplace_unstructured** executable in the /notebooks/C directory. The select OK > Next > Finished. The code will take a few seconds to finish profiling.

Take a moment to explore the profiler, and when you're ready, let's zoom in on the very beginning of our profile.

![unstructured_pgprof1.PNG](../images/unstructured_pgprof1.PNG)

We can see that we have uninterupted computation, and all of our data movement happens at the beginning of the program. This is ideal, because we are avoiding data transers in the middle of our computation. If you also profiled the structured version of the code, you will notice that the profiles are nearly identical. This isn't surprising, since the structured and unstructured approach work very similarly at the hardware level. However, structured data regions may be easier in simple codes, whereas some codes might flow better when using an unstructured approach. It is up to the programmer to decide which to use.

---

## OpenACC Update Directive

When we use the data directives, there exist two places where the programmer can transfer data between the host and the device. For the structured data directive we have the opportunity to transfer data at the beginning and at the end of the region. For the unstructured data directives, we can transfer data when we use the enter data and exit data directives.

However, there may be times in your program where you need to transfer data in the middle of a data region, or between an enter data and an exit data. In order to transfer data at those times, we can use the **update** directive. The update directive will explicitly transfer data between the host and the device. The **update** directive has two clauses:

**self**: The self clause will transfer data from the device to the host (GPU to CPU)  
**device**: The device clause will transfer data from the host to the device (CPU to GPU)

The syntax would look like:

**#pragma acc update self(A[0:N])**

**#pragma acc update device(A[0:N])**

All of the array shaping rules apply.

As an example, let's create a version of our laplace code where we want to print the array **A** after every 100 iterations of our loop. The code will look like this:

```
#pragma acc data copyin( A[:m*n],Anew[:m*n] )
{
    while ( error > tol && iter < iter_max )
    {
        error = calcNext(A, Anew, m, n);
        swap(A, Anew, m, n);
        
        if(iter % 100 == 0)
        {
            printf("%5d, %0.6f\n", iter, error);
            for( int i = 0; i < n; i++ )
            {
                for( int j = 0; j < m; j++ )
                {
                    printf("%0.2f ", A[i+j*m]);
                }
                printf("\n");
            }
        }
        
        iter++;

    }
}
```

Let's run this code (on a very small data set, so that we don't overload the console by printing thousands of numbers).


```sh
!cd update && pgcc -fast -ta=tesla:cc30 -Minfo=accel -o laplace_no_update jacobi.c laplace2d.c && ./laplace_no_update 10 10
```

We can see that the array is not changing. This is because the host copy of **A** is not being **updated** between loop iterations. Let's add the update directive, and see how the output changes.

```
#pragma acc data copyin( A[:m*n],Anew[:m*n] )
{
    while ( error > tol && iter < iter_max )
    {
        error = calcNext(A, Anew, m, n);
        swap(A, Anew, m, n);
        
        if(iter % 100 == 0)
        {
            printf("%5d, %0.6f\n", iter, error);
            
            #pragma acc update self(A[0:m*n])
            
            for( int i = 0; i < n; i++ )
            {
                for( int j = 0; j < m; j++ )
                {
                    printf("%0.2f ", A[i+j*m]);
                }
                printf("\n");
            }
        }
        
        iter++;

    }
}
```


```sh
!cd update/solution && pgcc -fast -ta=tesla:cc30 -Minfo=accel -o laplace_update jacobi.c laplace2d.c && ./laplace_update 10 10
```

---

## Conclusion

Relying on managed memory to handle data management can reduce the effort the programmer needs to parallelize their code, however, not all GPUs work with managed memory, and it is also lower performance than using explicit data management. OpenACC gives the programmer two main ways to handle data management, structured and unstructured data directives. By using these, the programmer is able to minimize the number of data transfers needed in their program.

---

## Bonus Task

If you would like some additional lessons on using OpenACC, there is an Introduction to OpenACC video series available from the OpenACC YouTube page. The fifth video in the series covers a lot of the content that was covered in this lab.  

[Introduction to Parallel Programming with OpenACC - Part 5](https://youtu.be/0zTX7-CPvV8)  

