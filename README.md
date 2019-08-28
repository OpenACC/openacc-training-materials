# OpenACC Official Training Materials
These training materials have been developed as a collaboration between the
University of Delaware and NVIDIA Corporation and are provided free of charge
by OpenACC.org. Please see [CONTRIBUTING.md] for license details.

## Modules

This material, which includes slides and code labs, is organized into modules
that are designed to build upon each other. Modules 1 - 6 are considered _core_
modules and modules 7 and beyond are considered advanced topics. 

The slides associated with these modules can be downloaded from
[https://drive.google.com/open?id=1d_eIwIRfScHxfJu6pnR28JrV3cMIwkIL]

The modules are organized as follows.

### Module 1 – Introduction to Parallel Programming
These modules are targeting students with potentially very different
backgrounds in OpenACC and parallel programming. As such, we do not assume that
the students are coming into these modules with any parallel programming
experience whatsoever. Module 1 is meant to be a fast paced “catch-up,” whereas
students are expected to gain a conceptual understanding of parallel
programming, and learn how to implement these concepts using OpenACC.

Topics that will be covered are as follows:
* Introduction to parallelism
* The goals of OpenACC
* Basic parallelization of code using OpenACC

Our focus on OpenACC in this module is very narrow; we introduce parallelism in
a very conceptual, graphical way, and show how we can relate these concepts to
simple OpenACC directives. The goal with this is to ensure that students
associate OpenACC with “general parallel programming” or “expressing
parallelism” rather than thinking that OpenACC is a “multicore/GPU programming
model.”

This module does not include a code lab.

### Module 2 – Profiling with OpenACC

These modules are meant to be profiler-driven. Students will make small changes
to their code, and re-profile it, comparing changes in the profiler vs.  their
expectations. [labs/module2](Module 2) will have students profile the
sequential Laplace Heat Equation code to obtain baseline results. Afterwards,
they will begin to implement a naïve parallelization of the code. The Laplace
Heat Equation code has 3 functions that are meant to be parallelized; students
will add OpenACC directives to each of these functions, one-by-one, and view
changes seen in the compiler. At this point, students are only expected to run
their code for a multicore accelerator. The GPU implementation is more complex,
and will be the focus of Module 3.

Topics that will be covered are as follows:
* Compiling sequential and OpenACC code
* The importance of code profiling
* Profiling sequential and OpenACC multicore code
* Technical introduction to the code used in introductory modules

This module will allow students to profile a sequential, and a multicore code
to observe the differences. The profiling process is mostly a step-by-step
tutorial, with regular pauses for explanation, or to allow students to explore
the features of the profiler. The code that students are profiling is the
Laplace code, which they will be working on for modules 2-6. We choose to focus
on multicore at first for two main reasons: multicore is a simpler platform to
program for, and we want to emphasis that OpenACC is more than a “GPU
programming model.” Students will begin working with GPUs in Module 4.

### Module 3 – Introduction to OpenACC Directives

In [labs/module3](Module 3) students will have already seen the parallel,
kernels, and loop directive at this point. Now, this module will focus on
teaching the specifics of each directive, the differences between them, and how
to use them to parallelize our code.

Topics that will be covered are as follows:
* The Parallel directive
* The Kernels directive
* The Loop directive

Students will learn the key differences between the parallel and kernels
directive, and can use either of them on the lab. It is recommended that they
try both directives, however. This module includes a lot of code examples, and
graphical representations of how the directives work with the hardware.
  
The lab section is designed for the students to achieve a working, near
optimal version of a multicore Laplace program.

### Module 4 – GPU Programming with OpenACC
  
[labs/module4](Module 4) is designed to teach students the key differences
between GPUs and multicore CPUs. We also delve into GPU memory management,
mostly from a conceptual level. We present CUDA Unified Memory as a reasonable
solution to memory management, and then finish the module with a guide to GPU
profiling using PGPROF. We also draw parallels between GPU architecture and our
OpenACC general parallel model.
  
The lab section will allow students to play with basic data clauses, and
managed memory. Then they will profile the code, and see how the changes they
are making things affect how the GPU is running (by viewing things such as time
spent on data transfers.)

Topics that will be covered are as follow:
* Definition of a GPU
* Basic OpenACC data management
* CUDA Unified Memory
* Profiling GPU applications

### Module 5 – Data Management with OpenACC

In Module 4, we introduced students to a very basic solution to GPU data
management. At the beginning of [labs/module5](Module 5), we highlight the
problems that this basic implementation has. The problem with our naïve
solution is that there is far too many data transfers between the compute
regions. Our program takes more time transferring data than it does computing
our answer. We will have students remedy this by using explicit data management
with the OpenACC data directive.

Topics that will be covered are as follows:
* OpenACC data directive/clauses
* OpenACC structured data region
* OpenACC unstructured data region
* OpenACC update directive
* Data management with C/C++ Structs/Classes

The bulk of this module will be code snippets and diagrams. We use diagrams to
represent CPU/GPU memory, and show the interaction between the two as we
analyze the data directive/clauses. The lab section will allow students to
experiment with both a structured and unstructured approach to data management
in their Laplace code.

### Module 6 – Loop Optimizations with OpenACC
  
[labs/module6](Module 6) is the last “core” module. After Module 6, we expect
students to be able to begin parallelizing their own personal code with OpenACC
with a good amount of confidence. The remaining modules after this point are
considered to be “advanced” modules, and are optional, and some may only be
applicable to specific audiences. Module 6 is all about loop clauses. This
module is meant be very visual, so that students can get a good sense of
exactly how each clause is affecting the execution of their loop.

Topics that will be covered are as follows:
* Seq/Auto clause
* Independent clause
* Reduction clause
* Collapse clause
* Tile clause
* Gang Worker Vector

This module touches on each of the loop clauses, show how they look within
code, and give a visual representation of it. The gang/worker/vector will most
likely be the lengthiest section in this module, just because it is the most
complex.  Also, in the lab section of Module 6, we will make our final
optimization to our Laplace code by utilizing loop optimizations and
gang/worker/vector. 

## Running the Docker container

The code labs have been written using Jupyter notebooks and a Dockerfile has
been built to simplify deployment. In order to serve the docker instance for a
student, it is necesary to expose port 8888 from the container, for instance,
the following command would expose port 8888 inside the container as port 8888
on the lab machine:

    $ docker run -rm --it -p 8888:8888 openacc-teaching-materials:latest

When this command is run, a student can browse to the serving machine on port
8888 using any web browser to access the labs. For instance, from if they are
running on the local machine the web browser should be pointed to
http://localhost:8888. The `-rm` flag is used to clean an temporary images
created during the running of the container and the `--it` flag enables killing
the jupyter server with `ctrl-c`. This command may be customized for your
hosting environment.

Modules 4 - 6 require an NVIDIA GPU to be run without customization.
