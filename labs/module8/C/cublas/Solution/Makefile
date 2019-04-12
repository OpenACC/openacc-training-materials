CC=pgcc
CUDAC=nvcc
CFLAGS=-ta=tesla:cc35 -Minfo=accel
CUDAFLAGS=-Mcuda

all: clean build
	./main

build: main.o ../matmult.o ../matmult_cublas.o
	$(CC) $(CFLAGS) -o main -lcublas main.o matmult.o matmult_cublas.o $(CUDAFLAGS)

main.o:
	$(CC) $(CFLAGS) -c main.c

matmult.o:
	$(CC) $(CFLAGS) -c matmult.c

matmult_cublas.o:
	$(CUDAC) -lcublas -c matmult_cublas.cu

run:
	./main

clean:
	-rm -f *.o main
