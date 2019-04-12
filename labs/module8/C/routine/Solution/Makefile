CC=pgcc
CUDAC=nvcc
CFLAGS=-ta=tesla:cc35 -Minfo=accel
CUDAFLAGS=-Mcuda

all: clean build
	./main

build: ../main.o ../distance_map.o distance_map_acc.o ../dist_cuda.o
	$(CC) $(CFLAGS) -o main dist_cuda.o main.o distance_map.o distance_map_acc.o $(CUDAFLAGS)

main.o:
	$(CC) $(CFLAGS) -c main.c

distance_map.o:
	$(CC) $(CFLAGS) -c distance_map.c

distance_map_acc.o:
	$(CC) $(CFLAGS) -c distance_map_acc.c

dist_cuda.o:
	$(CUDAC) -c -rdc=true dist_cuda.cu

run:
	./main

clean:
	-rm -f *.o main
