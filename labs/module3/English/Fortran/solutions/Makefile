.PHONY: all seq multicore kernels clean
all: seq multicore kernels

seq:
	make -C ./sequential

multicore:
	make -C ./multicore

kernels:
	make -C ./kernels

clean: 
	make clean -C ./sequential
	make clean -C ./multicore
	make clean -C ./kernels

