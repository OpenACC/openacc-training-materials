CC=pgcc
CUDAC=nvcc
CFLAGS=-ta=tesla:cc35 -Minfo=accel
CUDAFLAGS=-Mcuda

all: clean build
	./main

build: main.o ../dot.o ../dot_kernel.o
	$(CC) $(CFLAGS) -o main main.o dot.o dot_kernel.o $(CUDAFLAGS)

main.o:
	$(CC) $(CFLAGS) -c main.c

dot.o:
	$(CC) $(CFLAGS) -c dot.c

dot_kernel.o:
	$(CUDAC) -c dot_kernel.cu

run:
	./main

clean:
	-rm -f *.o main
