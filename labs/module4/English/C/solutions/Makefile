.PHONY: all seq multicore basic_data clean
all: seq multicore basic_data

seq:
	make -C ./sequential

multicore:
	make -C ./multicore

basic_data:
	make -C ./basic_data

clean: 
	make clean -C ./sequential
	make clean -C ./multicore
	make clean -C ./basic_data

