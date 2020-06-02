# Bonus Task (More about Gangs, Workers, and Vectors)

## Gang/Worker/Vector

This is our last optimization, and arguably the most important one. In OpenACC, **Gang Worker Vector** is used to define additional levels of parallelism. Specifically for NVIDIA GPUs, gang, worker, and vector will specify the *decomposition* of our loop iterations to GPU threads. Each loop will have an optimal Gang/Worker/Vector implementation, and finding that correct implementation will often take a bit of thinking, and possibly some trial and error. So let's explain how the `gang`, `worker`, and `vector` clauses actually work.

![gang_worker_vector.png](../images/gang_worker_vector.png)

This image represents a single **gang**. When parallelizing our **for loops**, the **loop iterations** will be **broken up evenly** among a number of gangs. Each gang will contain a number of **threads**. These threads are organized into **blocks**. A **worker** is a row of threads. In the above graphic, there are 3 **workers**, which means that there are 3 rows of threads. The **vector** refers to how long each row is. So in the above graphic, the vector is 8, because each row is 8 threads long.

By default, when programming for a GPU, **gang** and **vector** paralleism is automatically applied. Let's see a simple GPU sample code where we explicitly show how the gang and vector works.

```cpp
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

```cpp
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

```cpp
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

### Gang, Worker, and Vector Syntax

We have been showing really general examples of gang worker vector so far. One of the largest benefits of gang worker vector is the ability to explicitly define how many gangs and workers you need, and how many threads should be in the vector. Let's look at the syntax for the parallel directive:

```cpp
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

```cpp
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

### Avoid Wasting Threads

When parallelizing small arrays, you have to be careful that the number of threads within your vector is not larger than the number of loop iterations. Let's look at a simple example:

```cpp
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

```cpp
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

### The Rule of 32 (Warps)

The general rule of thumb for programming for NVIDIA GPUs is to always ensure that your vector length is a multiple of 32 (which means 32, 64, 96, 128, ... 512, ... 1024... etc.). This is because NVIDIA GPUs are optimized to use **warps**. Warps are groups of 32 threads that are executing the same computer instruction. So as a reference:

```cpp
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

```cpp
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

### Implementing the Gang Worker Vector

Use the following link to edit our code. Replace our ealier clauses with **gang, worker, and vector** To reorganize our thread blocks. Try it using a few different numbers, but always keep the vector length as a **multiple of 32** to fully utilize **warps**.

[laplace2d.c](laplace2d.c) 

(make sure to save your code with ctrl+s)

Then run the following script to see how the code runs.


```bash
$ pgcc -fast -ta=tesla -Minfo=accel -o laplace_gang_worker_vector jacobi.c laplace2d.c && ./laplace_gang_worker_vector
```

In our tests, it was difficult to beat our earlier code using `gang`, `worker`, and `vector` clauses, compared to the `tile` clause, but it is very common when optimizing real OpenACC codes to tweak loop mappings using these clauses and adjusting the vector length, so keep these clauses in the back of your mind for the future.
