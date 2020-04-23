# Data Management with OpenACC

This version of the lab is intended for Fortran programmers. The C/C++ version of this lab is available [here](../C/README.md).

You will receive a warning five minutes before the lab instance shuts down. Remember to save your work! If you are about to run out of time, please see the [Post-Lab](#Post-Lab-Summary) section for saving this lab to view offline later.

---
Let's execute the cell below to display information about the GPUs running on the server. To do this, execute the cell block below by giving it focus (clicking on it with your mouse), and hitting Ctrl-Enter, or pressing the play button in the toolbar above.  If all goes well, you should see some output returned below the grey cell.


```bash
$ pgaccelinfo
```

---

## Introduction

Our goal for this lab is to use the OpenACC Data Directives to properly manage our data.
  
<img src="../images/development_cycle.png" alt="OpenACC development cycle" width="50%">

This is the OpenACC 3-Step development cycle.

**Analyze** your code, and predict where potential parallelism can be uncovered. Use profiler to help understand what is happening in the code, and where parallelism may exist.

**Parallelize** your code, starting with the most time consuming parts. Focus on maintaining correct results from your program.

**Optimize** your code, focusing on maximizing performance. Performance may not increase all-at-once during early parallelization.

We are currently tackling the **parallelize** and **optimize** steps by adding the *data clauses* necessary to parallelize the code without CUDA Managed Memory and then *structured data directives* to optimize the data movement of our code.

---

## Run the Code (With Managed Memory)

In the [previous lab](../../../lab1/English/Fortran/README.md), we added OpenACC loop directives and relied on a feature called CUDA Managed Memory to deal with the separate CPU & GPU memories for us. Just adding OpenACC to our two loop nests we achieved a considerable performance boost. However, managed memory is not compatible with all GPUs or all compilers and it sometimes performs worse than programmer-defined memory management. Let's start with our solution from the previous lab and use this as our performance baseline. Note the runtime from the follow cell.


```bash
$ pgfortran -fast -ta=tesla,managed -Minfo=accel -o laplace_managed laplace2d.f90 jacobi.f90 && ./laplace_managed
```

### Optional: Analyze the Code

If you would like a refresher on the code files that we are working on, you may view both of them using the two links below.

[jacobi.f90](jacobi.f90)  
[laplace2d.f90](laplace2d.f90)  

## Building Without Managed Memory

Since we ultimately don't want to use CUDA Managed Memory, because it's less portable and often less performant than moving the data explicitly, let's removed the managed option from our compiler options. Try building and running the code now. What happens?


```bash
$ pgfortran -fast -ta=tesla -Minfo=accel -o laplace laplace2d.f90 jacobi.f90 && ./laplace
```

OK, so we're able to run, but we're running very slowly. We'll address that in just a moment, but first we should address an issue that you might encounter if you ever try this same exercise in C/C++. Because Fortran arrays contain all of the necessary size and shape information for the compiler to moved them to and from the device, this step works. Had you been programming in C/C++, however, the compiler would lack the information to move the arrays. Instead, you would have seen the following:

```
jacobi.c:
laplace2d.c:
PGC-S-0155-Compiler failed to translate accelerator region (see -Minfo messages): Could not find allocated-variable index for symbol (laplace2d.c: 47)
calcNext:
     47, Generating Tesla code
         48, #pragma acc loop gang /* blockIdx.x */
             Generating reduction(max:error)         50, #pragma acc loop vector(128) /* threadIdx.x */
     48, Accelerator restriction: size of the GPU copy of Anew,A is unknown
     50, Loop is parallelizable
PGC-F-0704-Compilation aborted due to previous errors. (laplace2d.c)
PGC/x86-64 Linux 19.10-0: compilation aborted
```

This error message is not very intuitive, so let me explain it to you.:

* `PGC-S-0155-Compiler failed to translate accelerator region (see -Minfo messages): Could not find allocated-variable index for symbol (laplace2d.c: 47)` - The compiler doesn't like something about a variable from line 47 of our code.
* `48, Accelerator restriction: size of the GPU copy of Anew,A is unknown` - I don't see any further information about line 47, but at line 48 the compiler is struggling to understand the size and shape of the arrays Anew and A. It turns out, this is our problem.

So, what these cryptic compiler errors are telling us is that the compiler needs to create copies of A and Anew on the GPU in order to run our code there, but it doesn't know how big they are, so it's giving up. We'll need to give the compiler more information about these arrays before it can move forward, so let's find out how to do that.

## OpenACC Data Clauses

Data clauses allow the programmer to specify data transfers between the host and device (or in our case, the CPU and the GPU). Because they are clauses, they can be added to other directives, such as the `parallel loop` directive that we used in the previous lab. Let's look at an example where we do not use a data clause.

```fortran
  allocate(A(N))

  !$acc parallel loop
  do i=1,100
    A(i) = 0
  enddo
```

We have allocated an array `A` outside of our parallel region. This means that `A` is allocated in the CPU memory. However, we access `A` inside of our loop, and that loop is contained within a *parallel region*. Within that parallel region, `A(i)` is attempting to access a memory location within the GPU memory. We didn't explicitly allocate `A` on the GPU, so one of two things will happen.

1. The compiler will understand what we are trying to do, and automatically copy `A` from the CPU to the GPU.
2. The program will check for an array `A` in GPU memory, it won't find it, and it will throw an error.

Instead of hoping that we have a compiler that can figure this out, we could instead use a **data clause**.

```fortran
  allocate(A(N))

  !$acc parallel loop copy(A(1:N))
  do i=1,100
    A(i) = 0
  enddo
```

We will learn the `copy` data clause first, because it is the easiest to use. With the inclusion of the `copy` data clause, our program will now copy the content of `A` from the CPU memory, into GPU memory. Then, during the execution of the loop, it will properly access `A` from the GPU memory. After the parallel region is finished, our program will copy `A` from the GPU memory back to the CPU memory. Let's look at one more direct example.

```cpp
  allocate(A(N))

  do i=1,100
    A(i) = 0
  enddo
  
  !$acc parallel loop copy(A(1:N))
  do i=1,100
    A(i) = 1
  enddo
```

Now we have two loops; the first loop will execute on the CPU (since it does not have an OpenACC parallel directive), and the second loop will execute on the GPU. Array `A` will be allocated on the CPU, and then the first loop will execute. This loop will set the contents of `A` to be all 0. Then the second loop is encountered; the program will copy the array `A` (which is full of 0's) into GPU memory. Then, we will execute the second loop on the GPU. This will edit the GPU's copy of `A` to be full of 1's.

At this point, we have two seperate copies of `A`. The CPU copy is full of 0's, and the GPU copy is full of 1's. Now, after the parallel region finishes, the program will copy `A` back from the GPU to the CPU. After this copy, both the CPU and the GPU will contain a copy of `A` that contains all 1's. The GPU copy of `A` will then be deallocated.

This image offers another step-by-step example of using the copy clause.

![copy_step_by_step](../images/copy_step_by_step_no_code.png)

We are also able to copy multiple arrays at once by using the following syntax.

```fortran
!$acc parallel loop copy(A[0:N], B[0:N])
do i = 1, N
    A(i) = B(i)
end do
```

Of course, we might not want to copy our data both to and from the GPU memory. Maybe we only need the array's values as inputs to the GPU region, or maybe it's only the final results we care about, or perhaps the array is only used temporarily on the GPU and we don't want to copy it either directive. The following OpenACC data clauses provide a bit more control than just the `copy` clause.

* `copyin` - Create space for the array and copy the input values of the array to the device. At the end of the region, the array is deleted without copying anything back to the host.
* `copyout` - Create space for the array on the device, but don't initialize it to anything. At the end of the region, copy the results back and then delete the device array.
* `create` - Create space of the array on the device, but do not copy anything to the device at the beginning of the region, nor back to the host at the end. The array will be deleted from the device at the end of the region.
* `present` - Don't do anything with these variables. I've put them on the device somewhere else, so just assume they're available.

You may also use them to operate on multiple arrays at once, by including those arrays as a comma separated list.

```fortran
!$acc parallel loop copy( A(1:N), B(1:M), C(1:Q) )
```

You may also use more than one data clause at a time.

```fortran
!$acc parallel loop create( A(1:N) ) copyin( B(1:M) ) copyout( C(1:Q) )
```

### Array Shaping

The shape of the array specifies how much data needs to be transferred. Let's look at an example:

```fortran
!$acc parallel loop copy(A(1:N))
do i = 1, N
  A(i) = 0
end do
```

Focusing specifically on the `copy(A(1:N))`, the shape of the array is defined within the brackets. The syntax for array shape is `(starting_index:ending_index)`. This means that (in the code example) we are copying data from array `A`, starting at index 1 (the start of the array), and copying until index N (which is most likely the length of the entire array).

We are also able to only copy a portion of the array:

```fortran
!$acc parallel loop copy(A(2:N-2))
```

This would copy all of the elements of `A` except for the first, and last element.

Lastly, if you do not specify a starting index, 1 is assumed. This means that

```fortran
!$acc parallel loop copy(A(1:N))
```

is equivalent to

```fortran
!$acc parallel loop copy(A(:N))
```

And since we're in Fortran, this can be shorted even more to just

```fortran
!$acc parallel loop copy(A(:))
```

## Making the Sample Code Work without Managed Memory

In order to build our example code without CUDA managed memory we need to give the compiler more information about the arrays. How do our two loop nests use the arrays `A` and `Anew`? The `calcNext` function take `A` as input and generates `Anew` as output, but also needs Anew copied in because we need to maintain that *hot* boundary at the top. So you will want to add a `copyin` clause for `A` and a `copy` clause for `Anew` on your region. The `swap` function takes `Anew` as input and `A` as output, so it needs the exact opposite data clauses. It's also necessary to tell the compiler the size of the two arrays by using array shaping. Our arrays are `m` times `n` in size, so we need to tell the compiler their shapes using the syntax above. Since we have already provided array shaping information in our `initialize` function using `allocate`, we can simply allow the compiler to use this information. Since our arrays are two dimensional, we can simply use `(:,:)` as our shape for both arrays. This tells the compiler to grab the existing shaping information from when these arrays were allocated. Go ahead and add data clauses to the two `parallel loop` directives in [laplace2d.f90](laplace2d.f90). Then try to build again.


```bash
$ pgfortran -fast -ta=tesla -Minfo=accel -o laplace laplace2d.f90 jacobi.f90 && ./laplace
```

Well, the good news is that it should have built correctly and run. If it didn't, check your data clauses carefully. The bad news is that now it runs a whole lot slower than it did before. Let's try to figure out why. The PGI compiler provides your executable with built-in timers, so let's start by enabling them and seeing what it shows. You can enable these timers by setting the environment variable `PGI_ACC_TIME=1`. Run the cell below to get the program output with the built-in profiler enabled.


```bash
$ pgfortran -fast -ta=tesla -Minfo=accel -o laplace laplace2d.f90 jacobi.f90 && PGI_ACC_TIME=1 ./laplace
```

Your output should look something like what you see below.
```
 completed in    182.941 seconds

Accelerator Kernel Timing data
/labs/lab2/English/Fortran/laplace2d.f90
  calcnext  NVIDIA  devicenum=0
    time(us): 53,265,706
    58: compute region reached 1000 times
        58: kernel launched 1000 times
            grid: [4094]  block: [128]
             device time(us): total=2,398,959 max=2,415 min=2,380 avg=2,398
            elapsed time(us): total=2,454,466 max=2,489 min=2,435 avg=2,454
        58: reduction kernel launched 1000 times
            grid: [2]  block: [256]
             device time(us): total=19,011 max=20 min=19 avg=19
            elapsed time(us): total=46,816 max=67 min=43 avg=46
    58: data region reached 4000 times
        58: data copyin transfers: 17000
             device time(us): total=33,881,820 max=2,141 min=6 avg=1,993
        66: data copyout transfers: 10000
             device time(us): total=16,965,916 max=2,135 min=9 avg=1,696
/labs/lab2/English/Fortran/laplace2d.f90
  swap  NVIDIA  devicenum=0
    time(us): 36,205,726
    76: compute region reached 1000 times
        76: kernel launched 1000 times
            grid: [4094]  block: [128]
             device time(us): total=2,306,221 max=2,319 min=2,293 avg=2,306
            elapsed time(us): total=2,363,356 max=2,397 min=2,347 avg=2,363
    76: data region reached 2000 times
        76: data copyin transfers: 8000
             device time(us): total=16,942,581 max=2,141 min=2,114 avg=2,117
        82: data copyout transfers: 9000
             device time(us): total=16,956,924 max=2,134 min=13 avg=1,884
```

 The total runtime was roughly 130 seconds with the profiler turned on (roughly 120 without). We can see that `calcNext` required roughly 53 seconds to run by looking at the `time(us)` line under the `calcNext` line. We can also look at the `data region` section and determine that 34 seconds were spent copying data to the device and 17 seconds copying data out for the device. The `swap` function has very similar numbers. That means that the program is actually spending very little of its runtime doing calculations. Why is the program copying so much data around? The screenshot below comes from the Nsight Systems profiler and shows part of one step of our outer while loop. The greenish and pink colors are data movement and the blue colors are our kernels (calcNext and swap). Notice that for each kernel we have copies to the device (greenish) before and copies from the device (pink) after. The means we have 4 segments of data copies for every iteration of the outer while loop.
 
![Profile before adding data region](../images/pre-data-f.png)
 
 Let's contrast this with the managed memory version. The image below shows the same program built with managed memory. Notice that there's a lot of "data migration" at the beginning, where the data is first used, but there's no data movement between the loops. This tells me that the data movement isn't really needed between these loops, but we need to tell the compiler that. 
 

![Profile using managed memory](../images/managed-f.png)

Because the loops are in two separate files, the compiler can't really see that the data is reused on the GPU between those function. We need to move our data movement up to a higher level where we can reuse it for each step through the program. To do that, we'll add OpenACC data directives.

---

## OpenACC Structured Data Directive

The OpenACC data directives allow the programmer to explicitly manage the data on the device (in our case, the GPU). Specifically, the structured data directive will mark a static region of our code as a **data region**.

```fortran
< Initialize data on host (CPU) >

!$acc data < data clauses >

    < Code >

!$acc end data
```

Device memory allocation happens at the beginning of the region, and device memory deallocation happens at the end of the region. Additionally, any data movement from the host to the device (CPU to GPU) happens at the beginning of the region, and any data movement from the device to the host (GPU to CPU) happens at the end of the region. Memory allocation/deallocation and data movement is defined by which clauses the programmer includes. This is a list of the most important data clauses that we can use:


### Encompassing Multiple Compute Regions

A single data region can contain any number of parallel/kernels regions. Take the following example:

```fortran
!$acc data copyin(A(1:N), B(1:N)) create(C(1:N))

    !$acc parallel loop
    do i = 1, N
        C(i) = A(i) + B(i)
    end do
    
    !$acc parallel loop
    do i = 1, N
        A(i) = C(i) + B(i)
    end do

!$acc end data
```

You may also encompass function calls within the data region:

```fortran
subroutine copy(A, B, N)

    !$acc parallel loop
    do i = 1, N
        A(i) = B(i)
    end do
    
end subroutine

...

!$acc data copyout(A(1:N),B(1:N)) copyin(C(1:N))

    call copy(A, C, N)
    
    call copy(A, B, N)
!$acc end data
```

### Adding the Structured Data Directive to our Code

Use the following links to edit our laplace code. Add a structured data directive to properly handle the arrays `A` and `Anew`. We've already added data clauses to our two functions, so this time we'll move up the calltree and add a structured data region around our while loop in the main program. Think about the input and output to this while loop and choose your data clauses for `A` and `Anew` accordingly.

[jacobi.f90](jacobi.f90)   

Then, run the following script to check you solution. You code should run just as good as (or slightly better) than our managed memory code.


```bash
$ pgfortran -fast -ta=tesla -Minfo=accel -o laplace laplace2d.f90 jacobi.f90 && ./laplace
```

Did your runtime go down? It should have but the answer should still match the previous runs. Let's take a look at the profiler now.

![Profile after adding data region](../images/post-data-f.png)

Notice that we no longer see the greenish and pink bars on either side of each iteration, like we did before. Instead, we see a red OpenACC `Enter Data` region which contains some greenish bars corresponding to host-to-device data transfer preceding any GPU kernel launches. This is because our data movement is now handled by the outer data region, not the data clauses on each loop. Data clauses count how many times an array has been placed into device memory and only copies data the outermost time it encounters an array. This means that the data clauses we added to our two functions are now used only for shaping and no data movement will actually occur here anymore, thanks to our outer `data` region.

This reference counting behavior is really handy for code development and testing. Just like we just did, you can add clauses to each of your OpenACC `parallel loop` or `kernels` regions to get everything running on the accelerator and then just wrap those functions with a data region when you're done and the data movement magically disappears. Furthermore, if you want to isolate one of those functions into a standalone test case you can do so easily, because the data clause is already in the code. 

---

## OpenACC Update Directive

When we use the data clauses you are only able to copy data between host and device memory at the beginning and end of your regions, but what if you need to copy data in the middle? For example, what if we wanted to debug our code by printing out the array every 100 steps to make sure it looks right. In order to transfer data at those times, we can use the `update` directive. The update directive will explicitly transfer data between the host and the device. The `update` directive has two clauses:

* `self` - The self clause will transfer data from the device to the host (GPU to CPU). You will sometimes see this clause called the `host` clause.
* `device` - The device clause will transfer data from the host to the device (CPU to GPU).

The syntax would look like:

`!$acc update self(A(1:N))`

`!$acc update device(A(1:N))`

All of the array shaping rules apply.

As an example, let's create a version of our laplace code where we want to print the array **A** after every 100 iterations of our loop. The code will look like this:

```fortran
!acc data copyin( A(n,m), Anew(n,m))

do while (error > tol && iter < iter_max)
    error = calcNext(A, Anew, m, n)
    swap(A, Anew, m, n)
    
    if(mod(iter,100).eq.0 ) then
      write(*,'(i5,f10.6)'), iter, error
      do i=1,n
        do j=1,m
          write(*,'(f10.2)', advance="no"), A(i,j)
        enddo
      enddo
    end if

    iter = iter+1
    
end do

!$acc end data
```

Let's run this code (on a very small data set, so that we don't overload the console by printing thousands of numbers). In order to do that you will have to edit [jacobi.f90](jacobi.f90) and change the dimensions of the problem from 4096 to 10. Otherwise the code will take a long time to run and it will produce a great deal of output to the screen.

**Once again**, please change the dimenions of the problem, `m` and `n` to 10. Originally the line in jacobi.f90 will be the following.

```fortran
integer, parameter :: n=4096, m=4096, iter_max=1000
```

That line in the jacobi.f90 now needs to look like the following
```fortran
integer, parameter :: n=10, m=10, iter_max=1000
```


```bash
$ cd update && pgfortran -fast -ta=tesla -Minfo=accel -o laplace_no_update laplace2d.f90 jacobi.f90 && ./laplace_no_update ; cd -
```

We can see that the array is not changing. This is because the host copy of `A` is not being **updated** between loop iterations. Let's add the update directive, and see how the output changes.

```fortran
!acc data copyin( A(n,m), Anew(n,m))

do while (error > tol && iter < iter_max)
    error = calcNext(A, Anew, m, n)
    swap(A, Anew, m, n)
      
    if(mod(iter,100).eq.0 ) then
      write(*,'(i5,f10.6)'), iter, error
      
      !acc update self(A(n,m))
      do i=1,n
        do j=1,m
          write(*,'(f10.2)', advance="no"), A(i,j)
        enddo
      enddo
    end if

    iter = iter+1
    
end do

!$acc end data
```


```bash
$ cd update/solution && pgfortran -fast -ta=tesla -Minfo=accel -o laplace_update laplace2d.f90 jacobi.f90 && ./laplace_update
```

Although you weren't required to add an `update` directive to this example code, except in the contrived example above, it's an extremely important directive for real applications because it allows you to do I/O or communication necessary for your code to execute without having to pay the cost of allocating and decallocating arrays on the device each time you do so.

---

## Conclusion

Relying on managed memory to handle data management can reduce the effort the programmer needs to parallelize their code, however, not all GPUs work with managed memory, and it is also lower performance than using explicit data management. In this lab you learned about how to use *data clauses* and *structured data directives* to explicitly manage device memory and remove your reliance on CUDA Managed Memory. 

---

## Bonus Task

If you would like some additional lessons on using OpenACC, there is an Introduction to OpenACC video series available from the OpenACC YouTube page. The fifth video in the series covers a lot of the content that was covered in this lab.  

[Introduction to Parallel Programming with OpenACC - Part 5](https://youtu.be/0zTX7-CPvV8)  

## Post-Lab Summary

You can execute the following cell block to create a zip-file of the files you've been working on.


```bash
%%bash
rm -f openacc_files.zip laplace laplace_managed *.o *.mod update/*.o update/*.mod update/solution/*.o update/solution/*.mod update/laplace_no_update update/solution/laplace_update
zip -r openacc_files.zip *
```
