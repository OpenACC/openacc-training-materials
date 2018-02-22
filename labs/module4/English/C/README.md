
# GPU Programming With OpenACC

Lab written by Eric Wright

This version of the lab is intended for C/C++ programmers. The Fortran version of this lab is available [here](../Fortran/OpenACC+GPU+Fortran.ipynb).

You will receive a warning five minutes before the lab instance shuts down. Remember to save your work! If you are about to run out of time, please see the [Post-Lab](#Post-Lab-Summary) section for saving this lab to view offline later.

---
Before we begin, let's verify [WebSockets](http://en.wikipedia.org/wiki/WebSocket) are working on your system.  To do this, execute the cell block below by giving it focus (clicking on it with your mouse), and hitting Ctrl-Enter, or pressing the play button in the toolbar above.  If all goes well, you should see some output returned below the grey cell.  If not, please consult the [Self-paced Lab Troubleshooting FAQ](https://developer.nvidia.com/self-paced-labs-faq#Troubleshooting) to debug the issue.


```python
print "The answer should be three: " + str(1+2)
```

Let's execute the cell below to display information about the GPUs running on the server.


```python
!nvidia-smi
```

---

## Introduction

Our goal for this lab is to learn how to run our code on a GPU (Graphical Processing Unit).
  
  
  
![development_cycle.png](../files/images/development_cycle.png)

This is the OpenACC 3-Step development cycle.

**Analyze** your code, and predict where potential parallelism can be uncovered. Use profiler to help understand what is happening in the code, and where parallelism may exist.

**Parallelize** your code, starting with the most time consuming parts. Focus on maintaining correct results from your program.

**Optimize** your code, focusing on maximizing performance. Performance may not increase all-at-once during early parallelization.

We are currently tackling the **parallelize** step. We have parallelized our code for a multicore CPU, and now we will learn what we need to do to get it running on a GPU.

---

## Run the Code (Multicore)

We have already completed a basic multicore implementation of our lab code. Run the following script IF you would prefer to use the parallel directive.


```python
!cp ./solutions/multicore/laplace2d.c ./laplace2d.c
```

---
If you would prefer to use the kernels directive, run the following script.


```python
!cp ./solutions/multicore/kernels/laplace2d.c ./laplace2d.c
```

---
Then you may run the multicore code by running the following script. An executable called **laplace_multicore** will be created.


```python
!pgcc -fast -ta=multicore -Minfo=accel -o laplace_multicore jacobi.c laplace2d.c && ./laplace_multicore
```

### Optional: Analyze the Code

If you would like a refresher on the code files that we are working on, you may view both of them using the two links below.

[jacobi.c](../../view/C/jacobi.c)  
[laplace2d.c](../../view/C/laplace2d.c)  

### Optional: Profile the Code

If you would like to profile the multicore code, you may select <a href="/vnc" target="_blank">this link.</a> This will open a noVNC window, then you may use PGPROF to profile our multicore laplace code. The executable will be found in the /notebooks/C directory.

---
## Optional: Introduction to GPUs (Graphical Processing Units)

GPUs were originally used to render computer graphics for video games. While they continue to dominate the video game hardware market, GPUs have also been adapted as a **high-throughput parallel hardware**. They excel at doing many things simultaneously.

![cpu_with_gpu.png](../files/images/cpu_with_gpu.png)

Similar to a multicore CPU, a GPU has multiple computational cores. A GPU will have many more cores, but these cores perform very badly when executing sequential serial code. Our goal when using a GPU is to only use it to offload our parallel code. All of our sequential code will continue to run on our CPU.

GPUs are what is known as a SIMD architecture (SIMD stands for: single instruction, multiple data). This means that GPUs excel at taking a single computer instruction (such as a mathematical instruction, or a memory read/write) and applying that instruction to a large amount of data. Ultimately, this means that a GPU can execute thousands of operations at the same time. This function is very similar to our multicore CPU architecture, except that with a GPU, we have a many more cores at our disposal.

![cpu_and_gpu_diagram.png](../files/images/cpu_and_gpu_diagram.png)

This diagram represents a machine that contains a CPU and a GPU. We can see that the CPU and GPU are two complete seperate devices, connected via an I/O Bus. This bus is traditionally a PCI-e bus, however, NVLink is a newer, faster alternative. These two devices also have seperate memory. This means that during the execution of our program, some amount of data will be transferred between the CPU and the GPU.

---
## Data Management With OpenACC

When programming for a GPU (or similar architecture), we must handle data management between the CPU and the GPU. The programmer is able to explicitly define data management by using the OpenACC **data directive and data clauses**. Otherwise, we are able to allow the copmpiler to handle all data management. Depending on the GPU (specifically older GPUs), allowing the compiler to handle data management might not be a viable option.

### Using OpenACC Data Clauses

Data clauses allow the programmer to specify data transfers between the host and device (or in our case, the CPU and the GPU). Let's look at an example where we do not use a data clause.

```
int *A = (int*) malloc(N * sizeof(int));

#pragma acc parallel loop
for( int i = 0; i < N; i++ )
{
    A[i] = 0;
}
```

We have allocated an array **A** outside of our parallel region. This means that **A** is allocated in the CPU memory. However, we access **A** inside of our loop, and that loop is contained within a **parallel region**. Within that parallel region, **A[i]** is attempting to access a memory location within the GPU memory. We didn't explicitly allocate **A** on the GPU, so one of two things will happen.

1. The compiler will understand what we are trying to do, and automatically copy **A** from the CPU to the GPU.
2. The program will check for an array **A** in GPU memory, it won't find it, and it will throw an error.

Instead of hoping that we have a compiler that can figure this out, we could instead use a **data clause**.

```
int *A = (int*) malloc(N * sizeof(int));

#pragma acc parallel loop copy(A[0:N])
for( int i = 0; i < N; i++ )
{
    A[i] = 0;
}
```

We will learn the **copy** data clause first, because it is the easiest to use. With the inclusion of the **copy** data clause, our program will now copy the content of **A** from the CPU memory, into GPU memory. Then, during the execution of the loop, it will properly access **A** from the GPU memory. After the parallel region is finished, our program will copy **A** from the GPU memory back to the CPU memory. Let's look at one more direct example.

```
int *A = (int*) malloc(N * sizeof(int));

for( int i = 0; i < N; i++ )
{
    A[i] = 0;
}

#pragma acc parallel loop copy(A[0:N])
for( int i = 0; i < N; i++ )
{
    A[i] = 1;
}
```

Now we have two loops; the first loop will execute on the CPU (since it does not have an OpenACC parallel directive), and the second loop will execute on the GPU. Array **A** will be allocated on the CPU, and then the first loop will execute. This loop will set the contents of **A** to be all 0. Then the second loop is encountered; the program will copy the array **A** (which is full of 0's) into GPU memory. Then, we will execute the second loop on the GPU. This will edit the GPU's copy of **A** to be full of 1's.

At this point, we have two seperate copies of **A**. The CPU copy is full of 0's, and the GPU copy is full of 1's. Now, after the parallel region finishes, the program will copy **A** back from the GPU to the CPU. After this copy, both the CPU and the GPU will contain a copy of **A** that contains all 1's. The GPU copy of **A** will then be deallocated.

This image offers another step-by-step example of using the copy clause.

![copy_step_by_step](../files/images/copy_step_by_step.png)

We are also able to copy multiple arrays at once by using the following syntax.

```
#pragma acc parallel loop copy(A[0:N], B[0:N])
for( int i = 0; i < N; i++ )
{
    A[i] = B[i];
}
```

### Array Shaping

The shape of the array specifies how much data needs to be transferred. Let's look at an example:

```
#pragma acc parallel loop copy(A[0:N])
for( int i = 0; i < N; i++ )
{
    A[i] = 0;
}
```

Focusing specifically on the **copy(A[0:N])**, the shape of the array is defined within the brackets. The syntax for array shape is **[starting_index:size]**. This means that (in the code example) we are copying data from array **A**, starting at index 0 (the start of the array), and copying N elements (which is most likely the length of the entire array).

We are also able to only copy a portion of the array:

```
#pragma acc parallel loop copy(A[1:N-2])
```

This would copy all of the elements of **A** except for the first, and last element.

Lastly, if you do not specify a starting index, 0 is assumed. This means that

```
#pragma acc parallel loop copy(A[0:N])
```

is equivalent to

```
#pragma acc parallel loop copy(A[:N])
```

### Including Data Clauses in our Laplace Code

Add **copy** data clause to our laplace code by selecting the following links:

[jacobi.c](../../edit/C/jacobi.c)  
[laplace2d.c](../../edit/C/laplace2d.c)  

Then, when you are ready, you may run the code by running the following script. It may not be intuitively obvious yet, but we are expecting the code to perform very poorly. For this reason, we are running our GPU code on a **significantly smaller input size**. If you were to run the GPU code on the full sized input, it will take several minutes to run.


```python
!pgcc -fast -ta=tesla:cc30 -Minfo=accel -o laplace_data_clauses jacobi.c laplace2d.c && ./laplace_data_clauses 1024 1024
```

If you are unsure about your answer, you can view the solution [here.](../../view/C/solutions/basic_data/laplace2d.c)

### Optional: Compiling GPU Code

Different GPUs will need to be compiled in slightly different ways. To get information about our GPU, we can use the **pgaccelinfo** command.


```python
!pgaccelinfo
```

There is a lot of information contained here, however, we are only going to focus on two points.

**Managed Memory:** will tell us whether or not our GPU supports CUDA managed memory. We will cover managed memory a little bit later in the lab.

**PGI Compiler Option:** tells us which target to compiler for. Ealier we were using a **-ta=multicore** flag for our multicore code. Now, to compile for our specific GPU, we will replace it with **-ta=tesla:cc30**.

---
### Profiling GPU Code

In order to understand why our program is performing so poorly, we should consult our profiler.

To profile our code, select <a href="/vnc" target="_blank">this link.</a> This will open a noVNC window, then you will use PGPROF to profile our edited laplace code.

To open a new profiling session, select File > New Session. You will be greeted with this window.

![pgprof1.PNG](../files/images/pgprof1.PNG)

Where it says "File: Enter executable file [required]", select Browse. Select File System > notebooks, then press OK.

![pgprof2.PNG](../files/images/pgprof2.PNG)

Open the C directory, and select the "laplace_data_clauses" executable.

![pgprof3.PNG](../files/images/pgprof3.PNG)

Then select OK. Your screen should look similar to this:

![pgprof4.PNG](../files/images/pgprof4.PNG)

As stated previously, if we run our program with the default 4096x4096 array, the program will take several minutes to run. I recommend that you reduce the size. Type "1024 1024" into cell labeled "Arguments", as pictured below.

![pgprof5.PNG](../files/images/pgprof5.PNG)

PGPROF will now profile your application. You will know when it is finished when you see the application output in the console window.

![pgprof6.PNG](../files/images/pgprof6.PNG)

This is the view that you should see once PGPROF is done profiling your program.

![pgprof7.PNG](../files/images/pgprof7.PNG)

We can see that our "timeline" has a lot going on. Feel free to explore the profile at this point. It will help to zoom in, so that you can better see the information.

![pgprof8.PNG](../files/images/pgprof8.PNG)

Upon zooming in, we get a much better idea of what is happening inside of our program. I have zoomed in on a single iteration of our while loop. We can see that both **calcNext** and **swap** is called. We can also see that there is a lot of space between them. It may be obvious now why our program is performing so poorly. The amount of time that our program is transferring data (as seen in the MemCpy timelines) is far greater than the time it takes running our computational functions **calcNext** and **swap**. In order to improve our performance, we need to minimize these data transfers.

---
## Managed Memory

![managed_memory.png](../files/images/managed_memory.png)  

As with many things in OpenACC, we have the option to allow the compiler to handle memory management. We will be able to achieve better performance by managing the memory ourselves, however, allowing the compiler to use managed memory is very simple, and will achieve much better performance than our naive solution from earlier. We do not need to make any changes to our code to get managed memory working. Simply run the following script. Keep in mind that unlike earlier, we are now running our code with the full sized 4096x4096 array.


```python
!pgcc -fast -ta=tesla:cc30,managed -Minfo=accel -o laplace_managed jacobi.c laplace2d.c && ./laplace_managed
```

### Optional: Compiling with the Managed Memory Flag

As long as the GPU supports managed memory (see [Optional: Compiling GPU Code](#Optional:-Compiling-GPU-Code) to learn how to check if your GPU supports it), all you need to do is add the managed option to our **-ta** flag.

**-ta=tesla:cc30,managed**

### Profiling our Managed Memory Code

Our program is doing a lot better. Let's re-profile it and see exactly why it improved. If you closed the noVNC window, you can reopen it by <a href="/vnc" target="_blank">this link.</a> To open a new profiling session, select File > New Session. Follow the steps from earlier, except select the "laplace_managed" executable. You may also emit the "Arguments". Once PGPROF is finished profiling our application, you should see the following view:  

![pgprof_managed1.PNG](../files/images/pgprof_managed1.PNG)

Feel free to explore the profile at this point. Then, when you're ready, let's zoom in.

![pgprof_managed3.PNG](../files/images/pgprof_managed3.PNG)

We can see that our compute regions (our **calcNext** and **swap** function calls) are much closer together now. There is significantly less data transfer happening between them. By using managed memory, the compiler was able to avoid the need to transfer data back and forth between the CPU and the GPU. In the next module, we will learn how to do this manually (which will boost the performance by a little bit), but for now, it is sufficient to use managed memory.

---

## Conclusion

We have learned how to run our code on a GPU using managed memory. We also experimented a little bit with managing the data ourselves, but that didn't work out as well as we had hoped. In the next module, we will expand on these data concepts and learn the proper way to manage our data, and will no longer need to rely on the compiler.

---

## Bonus Task

1. If you would like some additional lessons on using OpenACC to parallelize our code, there is an Introduction to OpenACC video series available from the OpenACC YouTube page. The third and fourth video in the series covers a lot of the content that was covered in this lab.  
[Introduction to Parallel Programming with OpenACC - Part 3](https://youtu.be/Pcc3O6h-YPE)  
[Introduction to Parallel Programming with OpenACC - Part 4](https://youtu.be/atXtVCHq8iw)

## Post-Lab Summary

If you would like to download this lab for later viewing, it is recommend you go to your browsers File menu (not the Jupyter notebook file menu) and save the complete web page.  This will ensure the images are copied down as well.

You can also execute the following cell block to create a zip-file of the files you've been working on, and download it with the link below.


```bash
%%bash
rm -f openacc_files.zip
zip -r openacc_files.zip /notebooks/C/*
```

**After** executing the above zip command, you should be able to download the zip file [here](files/openacc_files.zip)
