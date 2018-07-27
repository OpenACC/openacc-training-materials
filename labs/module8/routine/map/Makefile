CC=pgcc
CUDAC=nvcc
CFLAGS=-ta=tesla:cc35 -Minfo=accel
CUDAFLAGS=-Mcuda

build: main.o dot.o dot_acc.o map.o
	$(CC) $(CFLAGS) -o main main.o dot.o dot_acc.o map.o $(CUDAFLAGS)

main.o:
	$(CUDAC) -c main.cu

dot.o:
	$(CC) $(CFLAGS) -c dot.c

dot_acc.o:
	$(CC) $(CFLAGS) -c dot_acc.c

map.o:
	$(CC) $(CFLAGS) -c map.c

run:
	./main

clean:
	rm *.o main
