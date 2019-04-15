CC=pgcc
CFLAGS=-fast -ta=tesla -Minfo=accel
CXX=pgc++
CXXFLAGS=-std=c++11 -fast -ta=tesla -Minfo=accel
CUDAC=nvcc
CUDAFLAGS=

all: run

run: main
	./main

main: ../dist_cuda.o ../distance_map.o distance_map_acc.o ../main.o
	$(CC) $(CFLAGS) $^ -o $@ -Mcuda

.SUFFIXES: .o .cpp .c .cu
.cpp.o: 
	$(CXX) $(CXXFLAGS) -c -o $@ $<
.c.o: 
	$(CC) $(CFLAGS) -c -o $@ $<
.cu.o:
	$(CUDAC) $(CUDAFLAGS) -c -rdc=true -o $@ $<
.PHONY: clean
clean: 
	-rm -f *.o core main
