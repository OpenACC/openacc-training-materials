
# OpenACC Loop Optimizations

This version of the lab is intended for C/C++ programmers. The Fortran version of this lab is available [here](../Fortran/README.md).

---

## Introduction

Our goal for this lab is to use the OpenACC Loop clauses to opimize our Parallel Loops.
  
  
  
![development_cycle.png](../images/development_cycle.png)

This is the OpenACC 3-Step development cycle.

**Analyze** your code, and predict where potential parallelism can be uncovered. Use profiler to help understand what is happening in the code, and where parallelism may exist.

**Parallelize** your code, starting with the most time consuming parts. Focus on maintaining correct results from your program.

**Optimize** your code, focusing on maximizing performance. Performance may not increase all-at-once during early parallelization.

We are currently tackling the **optimize** step. We will include the OpenACC loop clauses to optimize the execution of our parallel loop nests.

---

## Run the Code

In the previous labs, we have built up a working parallel code that can run on both a multicore CPU and a GPU. Let's run the code and note the performance, so that we can compare the runtime to any future optimizations we make.


```sh
!pgcc -fast -ta=tesla:cc30 -Minfo=accel -o laplace_baseline jacobi.c laplace2d.c && ./laplace_baseline
```

### Optional: Analyze the Code

If you would like a refresher on the code files that we are working on, you may view both of them using the two links below.

[jacobi.c](jacobi.c)  
[laplace2d.c](laplace2d.c)  

### Optional: Profile the Code

If you would like to profile the code, you may select <a href="/vnc" target="_blank">this link.</a> This will open a noVNC window, then you may use PGPROF to profile our baseline laplace code. The laplace_baseline executable will be found in the /notebooks/C directory.

---

## OpenACC Loop Directive

The loop directive allows us to mark specific loops for parallelization. The loop directive also allows us to map specific optimizations/alterations to our loops using **loop clauses**. Not all loop clauses are used to optimize our code; some are also used to verify code correctness. A few examples of the loop directive are as follows:

```
#pragma acc parallel loop < loop clauses >
for(int i = 0; i < N; i++)
{
    < loop code >
}
```

```
#pragma acc kernels loop < loop clauses >
for(int i = 0; i < N; i++)
{
    < loop code >
}
```

```
#pragma acc parallel loop < loop clauses >
for(int i = 0; i < N; i++)
{
    #pragma acc loop < loop clauses >
    for(int j = 0; j < M; j++)
    {
        < loop code >
    }
}
```

Also, including loop optimizations does not always optimize the code. It is up to the programmer to decide which loop optimizations will work best for their loops.

### Independent Clause

When using the **kernels directive**, the compiler will decide which loops are, and are not, parallelizable. However, the programmer can override this compiler decision by using the **independent clause**. The independent clause is a way for the programmer to guarantee to the compiler that a loop is **parallelizable**.

```
#pragma acc kernels loop independent
for(int i = 0; i < N; i++)
{
    < Parallel Loop Code >
}



#pragma acc kernels
{
    for(int i = 0; i < N; i++)
    {
        < Parallel Loop Code >
    }
    
    #pragma acc loop independent
    for(int i = 0; i < N; i++)
    {
        < Parallel Loop Code >
    }
}
```

In the second example, we have two loops. The compiler will make a decision whether or not the first loop is parallelizable. In the second loop however, we have included the independent clause. This means that the compiler will trust the programmer, and assume that the second loop is parallelizable.

When using the **parallel directive**, the independent clause is automatically implied. This means that you do not need to use the **independent clause** when you are using the **parallel directive**.

### Auto Clause

The **auto clause** is more-or-less the complete opposite of the **independent clause**. When you are using the **parallel directive**, the compiler will trust anything that the programmer decides. This means that if the programmer believes that a loop is parallelizable, the compiler will trust the programmer. However, if you include the auto clause with your loops, the compiler will double check the loops, and decide whether or not to parallelize them.

```
#pragma acc parallel loop auto
for(int i = 0; i < N; i++)
{
    < Parallel Loop Code >
}
```

The **independent clause** is a way for the programmer to assert to the compiler that a loop is parallelizable. The **auto clause** is a way for the programmer to tell the compiler to analyze the loop, and to determine whether or not it is parallelizable.

### Seq Clause

The **seq clause** (short for "sequential") is used to define a loop that should run sequentially on the parallel hardware. This loop clause is usually automatically applied to large, multidimensional loop nests, since the compiler may only be able to describe parallelism for the outer-most loops. For example:

```
for(int i = 0; i < N; i++)
{
    for(int j = 0; j < M; j++)
    {
        for(int k = 0; k < Q; k++)
        {
            < Loop Code >
        }
    }
}
```

The compiler may only be able to parallelize the **i and j** loops, and will choose to run the **k** loop **sequentially**. The **seq clause** is also useful for running very small, nested loops sequentially. For example:

```
for(int i = 0; i < 1000000; i++)
{
    for(int j = 0; j < 4; j++)
    {
        for(int k = 0; k < 1000000; k++)
        {
            < Loop Code >
        }
    }
}
```

The middle loop is very small, and will most likely not benefit from parallelization. To fix this, we may apply the **seq clause** as follows:

```
#pragma acc parallel loop
for(int i = 0; i < 1000000; i++)
{
    #pragma acc loop seq
    for(int j = 0; j < 4; j++)
    {
        #pragma acc loop
        for(int k = 0; k < 1000000; k++)
        {
            < Loop Code >
        }
    }
}
```

In this code snippet, the middle loop will be run sequentially, while the outer-most loop and inner-most loop will be run in parallel.

### Reduction Clause

Up to this point, we have technically been using the **reduction clause** in our laplace code. We were not explicitly defining the reduction, instead the compiler has been automatically applying the reduction clause to our code. Let's look at one of the loops from within our laplace2d.c code file.

```
#pragma acc parallel loop present(A,Anew)
for( int j = 1; j < n-1; j++
{
    #pragma acc loop
    for( int i = 1; i < m-1; i++ )
    {
        Anew[OFFSET(j, i, m)] = 0.25 * ( A[OFFSET(j, i+1, m)] + A[OFFSET(j, i-1, m)] 
                                       + A[OFFSET(j-1, i, m)] + A[OFFSET(j+1, i, m)] );
        error = fmax( error, fabs(Anew[OFFSET(j, i, m)] - A[OFFSET(j, i , m)]));
    }
}
```

More specifically, let's focus on this single line of code:

```
error = fmax( error, fabs(Anew[OFFSET(j, i, m)] - A[OFFSET(j, i , m)]));
```

Each iteration of our inner-loop will write to the value **error**. When we are running thousands of these loop iterations **simultaneously**, it can become very dangerous to let all of them write directly to **error**. To fix this, we must use the OpenACC **reduction clause**. Let's look at the syntax.

```
#pragma acc parallel loop reduction(operator:value)
```

And let's look at a quick example of the use.

```
#pragma acc parallel loop reduction(+:sum)
for(int i = 0; i < N; i++)
{
    sum += A[i];
}
```

This is a list of all of the available operators in OpenACC.

|Operator    |Example                     |Description           |
|:----------:|:---------------------------|:---------------------|
|+           |reduction(+:sum)            |Mathematical summation|
|*           |reduction(*:product)        |Mathematical product  |
|max         |reduction(max:maximum)      |Maximum value         |
|min         |reduction(min:minimum)      |Minimum value         |
|&           |reduction(&:val)            |Bitwise AND           |
|&#124;      |reduction(&#124;:val)       |Bitwise OR            |
|&&          |reduction(&&:bool)          |Logical AND           |
|&#124;&#124;|reduction(&#124;&#124;:bool)|Logical OR            |

#### Optional: Implementing the Reduction Clause

We are compiling our code with the PGI compiler, which is automatically able to include the reduction clause. However, in other compilers, we may not be as fortunate. Use the following link to add the **reduction clause** with the **max operator** to our code.

[laplace2d.c](laplace2d.c)  
(make sure to save your code with ctrl+s)

You may then run the following script to verify that the compiler is properly recognizing your reduction clause.


```sh
!pgcc -ta=tesla:cc30 -Minfo=accel -o laplace_reduction jacobi.c laplace2d.c && ./laplace_reduction
```

You may also check your answer by selecting the following link.

[laplace2d.c](solutions/reduction/laplace2d.c)

### Private Clause

The private clause allows us to mark certain variables (and even arrays) as "private". The best way to visualize it is with an example:

```
int tmp;

#pragma acc parallel loop private(tmp)
for(int i = 0; i < N/2; i++)
{
    tmp = A[i];
    A[i] = A[N-i-1];
    A[N-i-1] = tmp;
}
```

In this code, each thread will have its own **private copy of tmp**. You may also declare static arrays as private, like this:

```
int tmp[10];

#pragma acc parallel loop private(tmp[0:10])
for(int i = 0; i < N; i++)
{
    < Loop code that uses the tmp array >
}
```

When using **private variables**, the variable only exists within the private scope. This generally means that the private variable only exists for a single loop iteration, and the values you store in the private variable cannot extend out of the loop.

### Collapse Clause

This is our first true **loop optimization**. The **collapse clause** allows us to transform a multi-dimensional loop nests into a single-dimensional loop. This process is helpful for increasing the overall length (which usually increases parallelism) of our loops, and will often help with memory locality. Let's look at the syntax.

```
#pragma acc parallel loop collapse( N )
```

Where N is the number of loops to collapse.

```
#pragma acc parallel loop collapse( 3 )
for(int i = 0; i < N; i++)
{
    for(int j = 0; j < M; j++)
    {
        for(int k = 0; k < Q; k++)
        {
            < loop code >
        }
    }
}
```

This code will combine the 3-dimensional loop nest into a single 1-dimensional loop. It is important to note that when using the **collapse clause**, the inner loops should not have their own **loop directive**. What this means is that the following code snippet is **incorrect** and will give a warning when compiling.

```
#pragma acc parallel loop collapse( 3 )
for(int i = 0; i < N; i++)
{
    #pragma acc loop
    for(int j = 0; j < M; j++)
    {
        #pragma acc loop
        for(int k = 0; k < Q; k++)
        {
            < loop code >
        }
    }
}
```

#### Implementing the Collapse Clause

Use the following link to edit our code. Use the **collapse clause** to collapse our multi-dimensional loops into a single dimensional loop.

[laplace2d.c](laplace2d.c)  
(make sure to save your code with ctrl+s)

Then run the following script to see how the code runs.


```sh
!pgcc -ta=tesla:cc30 -Minfo=accel -o laplace_collapse jacobi.c laplace2d.c && ./laplace_collapse
```

### Tile Clause

The **tile clause** allows us to break up a multi-dimensional loop into *tiles*, or *blocks*. This is often useful for increasing memory locality in some codes. Let's look at the syntax.

```
#pragma acc parallel loop tile( x, y, z, ... )
```

Our tiles can have as many dimensions as we want, though we must be careful to not create a tile that is too large. Let's look at an example:

```
#pragma acc parallel loop tile( 32, 32 )
for(int i = 0; i < N; i++)
{
    for(int j = 0; j < M; j++)
    {
        < loop code >
    }
}
```

The above code will break our loop iterations up into 32x32 tiles (or blocks), and then execute those blocks in parallel. Let's look at a slightly more specific code.

```
#pragma acc parallel loop tile( 32, 32 )
for(int i = 0; i < 128; i++)
{
    for(int j = 0; j < 128; j++)
    {
        < loop code >
    }
}
```

In this code, we have 128x128 loop iterations, which are being broken up into 32x32 tiles. This means that we will have 16 tiles, each tile being size 32x32. Similar to the **collapse clause**, the inner loops should not have the **loop directive**. This means that the following code is **incorrect** and will give a warning when compiling.

```
#pragma acc parallel loop tile( 32, 32 )
for(int i = 0; i < N; i++)
{
    #pragma acc loop
    for(int j = 0; j < M; j++)
    {
        < loop code >
    }
}
```

#### Implementing the Tile Clause

Use the following link to edit our code. Replace the**collapse clause** with the **tile clause** to break our multi-dimensional loops into smaller tiles. Try using a variety of different tile sizes, but always keep one of the dimensions as a **multiple of 32**. We will talk later about why this is important.

[laplace2d.c](laplace2d.c)  
(make sure to save your code with ctrl+s)

Then run the following script to see how the code runs.


```sh
!pgcc -ta=tesla:cc30 -Minfo=accel -o laplace_tile jacobi.c laplace2d.c && ./laplace_tile
```

### Gang/Worker/Vector

This is our last optimization, and arguably the most important one. In OpenACC, **Gang Worker Vector** is used to define additional levels of parallelism. Specifically for NVIDIA GPUs, gang worker vector will define the **organization** of our GPU threads. Each loop will have an optimal Gang Worker Vector implementation, and finding that correct implementation will often take a bit of thinking, and possibly some trial and error. So let's explain how Gang Worker Vector actually works.

![gang_worker_vector.png](../../images/gang_worker_vector.png)

This image represents a single **gang**. When parallelizing our **for loops**, the **loop iterations** will be **broken up evenly** among a number of gangs. Each gang will contain a number of **threads**. These threads are organized into **blocks**. A **worker** is a row of threads. In the above graphic, there are 3 **workers**, which means that there are 3 rows of threads. The **vector** refers to how long each row is. So in the above graphic, the vector is 8, because each row is 8 threads long.

By default, when programming for a GPU, **gang** and **vector** paralleism is automatically applied. Let's see a simple GPU sample code where we explicitly show how the gang and vector works.

```
#pragma acc parallel loop gang
for(int i = 0; i < N; i++)
{
    #pragma acc loop vector
    for(int j = 0; j < M; j++)
    {
        < loop code >
    }
}
```

The outer loop will be evenly spread across a number of **gangs**. Then, within those gangs, the inner-loop will be executed in parallel across the **vector**. This is a process that usually happens automatically, however, we can usually achieve better performance by optimzing the gang worker vector ourselves.

Lets look at an example where using gang worker vector can greatly increase a loops parallelism.

```
#pragma acc parallel loop gang
for(int i = 0; i < N; i++)
{
    #pragma acc loop vector
    for(int j = 0; j < M; k++)
    {
        for(int k = 0; k < Q; k++)
        {
            < loop code >
        }
    }
}
```

In this loop, we have **gang level** parallelism on the outer-loop, and **vector level** parallelism on the middle-loop. However, the inner-loop does not have any parallelism. This means that each thread will be running the inner-loop, however, GPU threads aren't really made to run entire loops. To fix this, we could use **worker level** parallelism to add another layer.

```
#pragma acc parallel loop gang
for(int i = 0; i < N; i++)
{
    #pragma acc loop worker
    for(int j = 0; j < M; k++)
    {
        #pragma acc loop vector
        for(int k = 0; k < Q; k++)
        {
            < loop code >
        }
    }
}
```

Now, the outer-loop will be split across the gangs, the middle-loop will be split across the workers, and the inner loop will be executed by the threads within the vector.

#### Gang Worker Vector Syntax

We have been showing really general examples of gang worker vector so far. One of the largest benefits of gang worker vector is the ability to explicitly define how many gangs and workers you need, and how many threads should be in the vector. Let's look at the syntax for the parallel directive:

```
#pragma acc parallel num_gangs( 2 ) num_workers( 4 ) vector_length( 32 )
{
    #pragma acc loop gang worker
    for(int i = 0; i < N; i++)
    {
        #pragma acc loop vector
        for(int j = 0; j < M; j++)
        {
            < loop code >
        }
    }
}
```

And now the syntax for the kernels directive:

```
#pragma acc kernels loop gang( 2 ) worker( 4 )
for(int i = 0; i < N; i++)
{
    #pragma acc loop vector( 32 )
    for(int j = 0; j < M; j++)
    {
        < loop code >
    }
}
```

#### Avoid Wasting Threads

When parallelizing small arrays, you have to be careful that the number of threads within your vector is not larger than the number of loop iterations. Let's look at a simple example:

```
#pragma acc kernels loop gang
for(int i = 0; i < 1000000000; i++)
{
    #pragma acc loop vector(256)
    for(int j = 0; j < 32; j++)
    {
        < loop code >
    }
}
```

In this code, we are parallelizing an inner-loop that has 32 iterations. However, our vector is 256 threads long. This means that when we run this code, we will have a lot more threads than loop iterations, and a lot of the threads will be sitting idly. We could fix this in a few different ways, but let's use **worker level parallelism** to fix it.

```
#pragma acc kernels loop gang worker(8)
for(int i = 0; i < 1000000000; i++)
{
    #pragma acc loop vector(32)
    for(int j = 0; j < 32; j++)
    {
        < loop code >
    }
}
```

Originally we had 1 (implied) worker, that contained 256 threads. Now, we have 8 workers that each have only 32 threads. We have eliminated all of our wasted threads by reducing the length of the **vector** and increasing the number of **workers**.

#### The Rule of 32 (Warps)

The general rule of thumb for programming for NVIDIA GPUs is to always ensure that your vector length is a multiple of 32 (which means 32, 64, 96, 128, ... 512, ... 1024... etc.). This is because NVIDIA GPUs are optimized to use **warps**. Warps are groups of 32 threads that are executing the same computer instruction. So as a reference:

```
#pragma acc kernels loop gang
for(int i = 0; i < N; i++)
{
    #pragma acc loop vector(32)
    for(int j = 0; j < M; j++)
    {
        < loop code >
    }
}
```

will perform much better than:

```
#pragma acc kernels loop gang
for(int i = 0; i < N; i++)
{
    #pragma acc loop vector(31)
    for(int j = 0; j < M; j++)
    {
        < loop code >
    }
}
```

#### Implementing the Gang Worker Vector

Use the following link to edit our code. Replace our ealier clauses with **gang, worker, and vector** To reorganize our thread blocks. Try it using a few different numbers, but always keep the vector length as a **multiple of 32** to fully utilize **warps**.

[laplace2d.c](laplace2d.c)  
(make sure to save your code with ctrl+s)

Then run the following script to see how the code runs.


```sh
!pgcc -ta=tesla:cc30 -Minfo=accel -o laplace_gang_worker_vector jacobi.c laplace2d.c && ./laplace_gang_worker_vector
```

## Using Everything we Learned

Now that we have covered the various ways to edit our loops, apply this knowledge to our laplace code. Try mixing some of the loop clauses, and see how the loop optimizations will differ between the parallel and the kernels directive.

You may run the following script to reset your code with the **kernels directive**.


```sh
!cp ./solutions/base_parallel/kernels/laplace2d.c ./laplace2d.c && echo "Reset Finished"
```

You may run the following script to reset your code with the **parallel directive**.


```sh
!cp ./solutions/base_parallel/parallel/laplace2d.c ./laplace2d.c && echo "Reset Finished"
```

Then use the following link to edit our laplace code.

[laplace2d.c](laplace2d.c)  
(make sure to save your code with ctrl+s)

Then run the following script to see how the code runs.


```sh
!pgcc -ta=tesla:cc30 -Minfo=accel -o laplace jacobi.c laplace2d.c && ./laplace
```

---

## Conclusion

Our primary goal when using OpenACC is to parallelize our large for loops. To accomplish this, we must use the OpenACC loop directive and loop clauses. There are many ways to alter and optimize our loops, though it is up to the programmer to decide which route is the best to take. At this point in the lab series, you should be able to begin parallelizing your own personal code, and to be able to achieve a relatively high performance using OpenACC.

---

## Bonus Task

If you would like some additional lessons on using OpenACC, there is an Introduction to OpenACC video series available from the OpenACC YouTube page. If you haven't already, I recommend watching this 6 part series. Each video is under 10 minutes, and will give a visual, and hands-on look at a lot of the material we have covered in these labs. The following link will bring you to Part 1 of the series.

[Introduction to Parallel Programming with OpenACC - Part 1](https://youtu.be/PxmvTsrCTZg)  

