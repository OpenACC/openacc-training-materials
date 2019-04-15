CC=pgcc
CFLAGS=-fast -ta=tesla -Minfo=accel
CXX=pgc++
CXXFLAGS=-std=c++11 -fast -ta=tesla -Minfo=accel
CUDAC=nvcc
CUDAFLAGS=

all: run

run: main
	./main

main: dot_acc.o ../dot.o ../main.o
	$(CC) $(CFLAGS) $^ -o $@ -Mcuda

.SUFFIXES: .o .cpp .c .cu
.cpp.o: 
	$(CXX) $(CXXFLAGS) -c -o $@ $<
.c.o: 
	$(CC) $(CFLAGS) -c -o $@ $<
.cu.o:
	$(CUDAC) $(CUDAFLAGS) -c -o $@ $<
.PHONY: clean
clean: 
	-rm -f *.o core main
