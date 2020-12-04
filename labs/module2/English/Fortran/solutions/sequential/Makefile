CC=pgcc
CXX=pgc++
FC=pgfortran
LD=pgfortran
CFLAGS=-fast -ta=tesla 
FFLAGS=-fast -ta=tesla 
CXXFLAGS=-fast -ta=tesla 
LDFLAGS=-fast -ta=tesla  

NVTXINC := -I/opt/nvidia/hpc_sdk/Linux_x86_64/20.9/cuda/11.0/include -lnvToolsExt 
NVTXLIB := -L/opt/nvidia/hpc_sdk/Linux_x86_64/20.9/cuda/11.0/lib64/

all: exe run

exe: nvtx.o laplace2d.o jacobi.o
	$(LD) $(LDFLAGS) -o $@ $^  $(NVTXLIB) $(NVTXINC)
 
run: exe
	./exe

.PHONY: clean
clean:
	-rm -f *.o *.mod core exe
.SUFFIXES: .c .cpp .F90 .f90 .o
.cpp.o:
	$(CXX) $(CXXFLAGS) -c -o $@ $<  $(NVTXLIB) $(NVTXINC)
.c.o:
	$(CC) $(CFLAGS) -c -o $@ $<  $(NVTXLIB) $(NVTXINC)
.F90.o:
	$(FC) $(FFLAGS) -c -o $@ $<  $(NVTXLIB) $(NVTXINC)
.f90.o:
	$(FC) $(FFLAGS) -c -o $@ $<  $(NVTXLIB) $(NVTXINC)
