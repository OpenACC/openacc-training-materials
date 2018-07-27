CC=pgcc
CUDAC=nvcc
CFLAGS=-ta=tesla:cc35,rdc,cuda9.1 -Minfo=accel
CUDAFLAGS=-Mcuda

build: main.o distance_map.o distance_map_acc.o dist_cuda.o
	$(CC) $(CFLAGS) -o main main.o distance_map.o distance_map_acc.o dist_cuda.o $(CUDAFLAGS)

main.o:
	$(CC) $(CFLAGS) -c main.c

distance_map.o:
	$(CC) $(CFLAGS) -c distance_map.c

distance_map_acc.o:
	$(CC) $(CFLAGS) -c distance_map_acc.c

dist_cuda.o:
	$(CUDAC) -c -rdc true dist_cuda.cu

run:
	./main

clean:
	rm *.o main
