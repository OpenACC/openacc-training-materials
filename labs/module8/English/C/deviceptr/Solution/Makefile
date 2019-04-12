CC=pgcc
CUDAC=nvcc
CFLAGS=-ta=tesla:cc35 -Minfo=accel
CUDAFLAGS=-Mcuda

all: clean build
	./main

build: ../main.o ../dot.o dot_acc.o
	$(CC) $(CFLAGS) -o main main.o dot.o dot_acc.o $(CUDAFLAGS)

main.o:
	$(CUDAC) -c main.cu

dot.o:
	$(CC) $(CFLAGS) -c dot.c

dot_acc.o:
	$(CC) $(CFLAGS) -c dot_acc.c

run:
	./main

clean:
	-rm -f *.o main
