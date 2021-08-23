# Profiling OpenACC Code

This lab is intended for Fortran programmers. If you prefer to use C/C++, click [this link.](../C/README.md)

The following timer counts down to a five minute warning before the lab instance shuts down.  You should get a pop up at the five minute warning reminding you to save your work!  If you are about to run out of time, please see the [Post-Lab](#Post-Lab-Summary) section for saving this lab to view offline later.

Don't forget to check out additional [OpenACC Resources](https://www.openacc.org/resources) and join our [OpenACC Slack Channel](https://www.openacc.org/community#slack) to share your experience and get more help from the community.

---
Let's execute the cell below to display information about the GPUs running on the server by running the `pgaccelinfo` command, which ships with the PGI compiler that we will be using. To do this, execute the cell block below by giving it focus (clicking on it with your mouse), and hitting Ctrl-Enter, or pressing the play button in the toolbar above.  If all goes well, you should see some output returned below the grey cell.


```bash
$ pgaccelinfo
```

---
## Introduction

Our goal for this lab is to learn what exactly code profiling is, and how we can use it to help us write powerful parallel programs.  
  
  
  
![development-cycle.png](../images/development-cycle.png)

This is the OpenACC 3-Step development cycle.

**Analyze** your code to determine most likely places needing parallelization or optimization.

**Parallelize** your code by starting with the most time consuming parts and check for correctness.

**Optimize** your code to improve observed speed-up from parallelization.

We are currently tackling the **analyze** step. We will use PGI's code profiler (PGProf) to get an understanding of a relatively simple sample code before moving onto the next two steps.

## Run the Code

Our first step to analyzing this code is to run it. We need to record the results of our program before making any changes so that we can compare them to the results from the parallel code later on. It is also important to record the time that the program takes to run, as this will be our primary indicator to whether or not our parallelization is improving performance.

### Compiling Code with PGI

We are using the PGI compiler to compiler our code. You will not need to memorize the compiler commands to complete this lab, however, they will be helpful to know if you want to parallelize your own personal code with OpenACC.

**pgcc**      : this is the command to compile C code  
**pgc++**     : this is the command to compile C++ code  
**pgfortran** : this is the command to compile Fortran code  
**-fast**     : this compiler flag will allow the compiler to perform additional optimizations to our code


```bash
$ pgfortran -fast -o laplace laplace2d.f90 jacobi.f90 && echo "Compilation Successful!" && ./laplace
```

### Understanding Code Results

The output from our program will make more sense as we analyze the code. The most important thing to keep in mind is that we need these output values to stay consistent. If these outputs change during any point while we parallelize our code, we know we've made a mistake. For simplicity, focus on the last output, which occurred at iteration 900. It is also helpful to record the time the program took to run. Our goal while parallelizing the code is ultimately to make it faster, so we need to know our "base runtime" in order to know if the code is running faster.

## Analyze the Code

Now that we know how long the code took to run, and what the code's output looks like, we should be able to view the code with a decent idea of what is happening. The code is contained within two files, which you may open and view.

[jacobi.f90](../Fortran/jacobi.f90)  
[laplace2d.f90](../Fortran/laplace2d.f90)  
  
You may read through these two files on your own, but we will also highlight the most important parts below in the "Code Breakdown".

### Code Description

The code simulates heat distribution across a 2-dimensional metal plate. In the beginning, the plate will be unheated, meaning that the entire plate will be room temperature. Then, a constant heat will be applied to the edge of the plate, then the code will simulate that heat distributing across the plate.  

This is a visual representation of the plate before the simulation starts:  
  
![plate1.png](../images/plate1.png)  
  
We can see that the plate is uniformly room temperature, except for the top edge. Within the [laplace2d.f90](../Fortran/laplace2d.f90) file, we see a subroutine called `initialize`. This function is what "heats" the top edge of the plate. 
  
```fortran
    subroutine initialize(A, Anew, m, n)
      integer, parameter :: fp_kind=kind(1.0d0)
      real(fp_kind),allocatable,intent(out)   :: A(:,:)
      real(fp_kind),allocatable,intent(out)   :: Anew(:,:)
	  integer,intent(in)          :: m, n

      allocate ( A(0:n-1,0:m-1), Anew(0:n-1,0:m-1) )

      A    = 0.0_fp_kind
      Anew = 0.0_fp_kind

      A(0,:)    = 1.0_fp_kind
      Anew(0,:) = 1.0_fp_kind
    end subroutine initialize
```

After the top edge is heated, the code will simulate the heat distributing across the length of the plate. We will keep the top edge at a constant heat as the simulation progresses.

This is the plate after several iterations of our simulation:  
  
![plate2.png](../images/plate2.png) 

That's the theory: simple heat distribution. However, we are more interested in how the code works. 

### Code Breakdown

The 2-dimensional plate is represented by a 2-dimensional array containing double values. These doubles represent temperature; 0.0 is room temperature, and 1.0 is our max temperature. The 2-dimensional plate has two states, one represents the current temperature, and one represents the expected temperature values at the next step in our simulation. These two states are represented by arrays **`A`** and **`Anew`** respectively. The following is a visual representation of these arrays, with the top edge "heated".

![plate_sim2.png](../images/plate_sim2.png)  
    
Simulating this state in two arrays is very important for our `calcNext` function. Our calcNext is essentially our "simulate" function. calcNext will look at the inner elements of A (meaning everything except for the edges of the plate) and update each elements temperature based on the temperature of its neighbors.  If we attempted to calculate in-place (using only `A`), then each element would calculate its new temperature based on the updated temperature of previous elements. This data dependency not only prevents parallelizing the code, but would also result in incorrect results when run in serial. By calculating into the temporary array `Anew` we ensure that an entire step of our simulation has completed before updating the `A` array.

![plate_sim3.png](../images/plate_sim3.png)  

This is the `calcNext` function:

```fortran
01 function calcNext(A, Anew, m, n)
02   integer, parameter          :: fp_kind=kind(1.0d0)
03   real(fp_kind),intent(inout) :: A(0:n-1,0:m-1)
04   real(fp_kind),intent(inout) :: Anew(0:n-1,0:m-1)
05   integer,intent(in)          :: m, n
06   integer                     :: i, j
07   real(fp_kind)               :: error
08	  
09   error=0.0_fp_kind
10	  
11   do j=1,m-2
12     do i=1,n-2
13        Anew(i,j) = 0.25_fp_kind * ( A(i+1,j  ) + A(i-1,j  ) + &
14                                     A(i  ,j-1) + A(i  ,j+1) )
15        error = max( error, abs(Anew(i,j)-A(i,j)) )
16     end do
17   end do
18   calcNext = error
19 end function calcNext
```

We see on lines 13 and 14 where we are calculating the value of `Anew` at `i,j` by averaging the current values of its neighbors. Line 09 is where we calculate the current rate of change for the simulation by looking at how much the `i,j` element changed during this step and finding the maximum value for this `error`. This allows us to short-circuit our simulation if it reaches a steady state before we've completed our maximum number of iterations.

Lastly, our `swap` subroutine will copy the contents of `Anew` to `A`.

```fortran
01 subroutine swap(A, Anew, m, n)
02   integer, parameter        :: fp_kind=kind(1.0d0)
03   real(fp_kind),intent(out) :: A(0:n-1,0:m-1)
04   real(fp_kind),intent(in)  :: Anew(0:n-1,0:m-1)
05   integer,intent(in)        :: m, n
06   integer                   :: i, j
07 
08   do j=1,m-2
09     do i=1,n-2
10       A(i,j) = Anew(i,j)
11     end do
12   end do
13 end subroutine swap
```


## Profile the Code

By now you should have a good idea of what the code is doing. If not, go spend a little more time in the previous sections to ensure you understand the code before moving forward. Now it's time to profile the code to get a better understanding of where the application is spending its runtime. To profile our code we will be using PGPROF. PGPROF provides both command line and visual profiler. It comes comes with the PGI compiler. 

We will start by profiling the laplace executable that we created earlier using the command line option first. Run the pgprof command: 


```bash
$ pgprof ./laplace
```

We can see the time that each individual portion of our code took to run. This information is important because it allows us to make educated decisions about which parts of our code to optimize first. To get the bang for our buck, we want to focus on the most time-consuming parts of the code. Next, we will compiler, run, and profile a parallel version of the code, and analyze the differences.

### Optional - Where is the __c_mcopy8 coming from?

When we compiled our code earlier, we omitted any sort of compiler feedback. It turns out that even with a sequential code, the compiler is performing a lot of optimizations. If you compile the code again with the `-Minfo=opt` flag, which instructs  the compiler to print additional information how it optimized the code, then it will become more obvious where this strange routine came from. Afterwards, you should see that the `__c_mcopy8` is actually an optimzation that is being applied to the **`swap`** function. Notice in the output below that at line 64 of `laplace2d.c`, which happens inside the `swap` routine, that the compiler determined that our loops are performing a memory copy, which it believes can be performed more efficiently by calling the `__c_mcopy8` function instead.

```
laplace2d.f90:
swap:
     76, Memory copy idiom, loop replaced by call to __c_mcopy8
```


```bash
$ pgfortran -fast -Minfo=opt -o laplace laplace2d.f90 jacobi.f90
```

## Run Our Parallel Code on Multicore CPU

In a future lab you will run parallelize the code to run on a multicore CPU. This is the simplest starting point, since it doesn't require us to think about copying our data between different memories. So that you can experience profiling with PGProf on a multicore CPU, a parallel version of the code has been provided. You will be able to parallelize the code yourself in the next lab.


```bash
$ pgfortran -fast -ta=multicore -Minfo=accel -o laplace_parallel ./solutions/parallel/laplace2d.f90 ./solutions/parallel/jacobi.f90 && ./laplace_parallel
```

### Compiling Multicore Code using PGI

Again, you do not need to memorize the compiler commands to complete this lab. Though, if you want to use OpenACC with your own personal code, you will want to learn them.

**-ta** : This flag will tell the compiler to compile our code for a specific parallel hardware. TA stands for *"Target Accelerator"*, an accelerator being any device that accelerates performance (in our case, this means parallel hardware.) Omitting the -ta flag will cause the code to compile sequentially.  
**-ta=multicore** will tell the compiler to parallelize the code specifically for a multicore CPU.  
**-Minfo** : This flag will tell the compiler to give us some feedback when compiling the code.  
**-Minfo=accel** : will only give us feedback about the parallelization of our code.  
**-Minfo=opt** : will give us feedback about sequential optimizations.  
**-Minfo=all** : will give all feedback; this includes feedback about parallelizaton, sequential optimizations, and even parts of the code that couldn't be optimized for one reason or another.  

If you would like to see the c_mcopy8 from earlier, try switching the Minfo flag with `-Minfo=accel,opt`.


## Profiling Multicore Code

We will use PGProf visual profiler this time to get a more graphical view of the profile. [Click here](/vnc) to open a new browser tab with a virtual desktop runnning the PGProf profiler. Normally you would open this program on your local machine by running the `pgprof` command or choosing it from your installed applications.

After accessing the URL, the first screen will ask you to connect to VNC ![connect.png](../images/connect.png)
You will be asked to enter the password. Please enter "nvidia" as the password ![password.png](../images/password.png)
We have already run Visual Profiler for you and the first screen you see is to choose a workspace. You can keep the default workspace and press enter ![workspace.png](../images/workspace.png)

We will start by profiling the laplace executable that we created earlier. To do this, Select File > New Session. After doing this you should see a pop-up like the one in the picture below.

![pgprof1.png](../images/pgprof1.png) 

Then where is says "File: Enter Executable File [required]", select "Browse". Then select File Systems > home > openacc > labs > module2 > English > Fortran.

Select our "laplace" executable file. 

![location.png](../images/location.png)

Then select "Next", followed by "Finished". 

![pgprof4.png](../images/pgprof4.PNG) 


Follow the steps from earlier to profile the code with PGPROF, however, select the **`laplace_parallel`** executable this time instead of **`laplace`**. If you have closed the noVNC client, you can reopen it by <a href="/vnc" target="_blank">clicking this link</a>.

This is the view that we are greeted with when profiling a multicore application.

![pgprof_parallel1.PNG](../images/pgprof_parallel1.PNG)

The first difference we see is the blue "timeline." This timeline represents when our program is executing something on the parallel hardware. This means that every call to `calcNext` and `swap` should be represented by a blue bar. 

This is the main idea behind parallel programming. When you consider the **TOTAL** computation time, it will take almost equivalent time to our sequential program, however the application runtime decreases due to the fact that we can now execute portions of our code in parallel by spreading the work across multiple threads.

## Conclusion

Now we have a good understanding of how our program is running, and which parts of the program are time consuming. In the next lab, we will parallelize our program using OpenACC.

We are working on a very simple code that is specifically used for teaching purposes. Meaning that, in terms of complexity, it can be fairly underwhelming. Profiling code will become exponentially more useful if you chose to work on a "real-world" code; a code with possibly hundreds of functions, or millions of lines of code. Profiling may seem trivial when we only have 4 functions, and our entire code is contained in only two files, however, profiling will be one of your greatest assets when parallelizing real-world code.

## Bonus Task

For right now, we are focusing on multicore CPUs. Eventually, we will transition to GPUs. If you are familiar with GPUs, and would like to play with a GPU profile, then feel free to try this bonus task. If you do not want to complete this task now, you will have an opportunity in later labs (where we will also explain more about what is happening.)

Run this script to compile/run our code on a GPU.


```bash
$ pgfortran -fast -ta=tesla -Minfo=accel -o laplace_gpu ./solutions/gpu/laplace2d.f90 ./solutions/gpu/jacobi.f90 && ./laplace_gpu
```

Now, within PGPROF, select File > New Session. Follow the same steps as earlier, except select the **`laplace_gpu`** executable. If you close the noVNC window, you can reopen it by <a href="/vnc" target="_blank">clicking this link</a>.

Happy profiling!

## Post-Lab Summary

If you would like to download this lab for later viewing, it is recommend you go to your browsers File menu (not the Jupyter notebook file menu) and save the complete web page.  This will ensure the images are copied down as well.

You can also execute the following cell block to create a zip-file of the files you've been working on, and download it with the link below.


```python
%%bash
rm -f openacc_files.zip
zip -r openacc_files.zip ../C/* ../Fortran/*
```

**After** executing the above zip command, you should be able to download the zip file [here](files/openacc_files.zip)

# Licensing
This material is released by NVIDIA Corporation under the Creative Commons Attribution 4.0 International (CC BY 4.0).
