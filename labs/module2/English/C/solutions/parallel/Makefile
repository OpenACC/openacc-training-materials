CC=pgcc
CXX=pgc++
FC=pgfortran
LD=pgcc
CFLAGS=-fast -ta=tesla:managed 
FFLAGS=-fast -ta=tesla:managed 
CXXFLAGS=-fast -ta=tesla:managed 
LDFLAGS=-fast -ta=tesla:managed 

NVTXLIB := -I/opt/nvidia/hpc_sdk/Linux_x86_64/20.9/cuda/11.0/include

all: exe run

exe: laplace2d.o jacobi.o
	$(LD) $(LDFLAGS) -o $@ $^ ${NVTXLIB} 

run: exe
	./exe

.PHONY: clean
clean:
	-rm -f *.o *.mod core exe
.SUFFIXES: .c .cpp .F90 .o
.cpp.o:
	$(CXX) $(CXXFLAGS) -c -o $@ $< ${NVTXLIB} 
.c.o:
	$(CC) $(CFLAGS) -c -o $@ $< ${NVTXLIB} 
.F90.o:
	$(FC) $(FFLAGS) -c -o $@ $< ${NVTXLIB} 
