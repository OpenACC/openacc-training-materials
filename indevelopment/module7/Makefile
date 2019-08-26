CC=pgcc
CFLAGS=-g -fast -ta=tesla:cc70,pinned -Minfo=accel
CXX=pgc++
CXXFLAGS=-g -fast -ta=tesla:cc70,pinned -Minfo=accel
FC=pgfortran
FCFLAGS=-g -fast -ta=tesla:cc70,pinned -Minfo=accel

all: filter.x filter_f90.x

filter.x: imageWrapper.o filter.o main.o
	$(CXX) $(CXXFLAGS) $^ -lopencv_highgui -lopencv_imgcodecs -lopencv_core -lm -o $@ -L$(CUDA_HOME)/lib64 -lcudart
filter_f90.x: imageWrapper.o filter_f90.o main_f90.o
	$(CXX) $(CXXFLAGS) $^ -lopencv_highgui -lopencv_imgcodecs -lopencv_core -lm -o $@ -L$(CUDA_HOME)/lib64 -lcudart -pgf90libs

MAKEFLAGS += --no-builtin-rules
.SUFFIXES:
.SUFFIXES: .o .cpp .c .F90 
.cpp.o: 
	$(CXX) $(CXXFLAGS) -c -o $@ $<
.c.o: 
	$(CC) $(CFLAGS) -c -o $@ $<
.F90.o:
	$(FC) $(FCFLAGS) -c -o $@ $<
.PHONY: clean
clean: 
	-rm -f *.o *.mod core filter.x filter_f90.x
