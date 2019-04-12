.PHONY: all seq multicore gpu clean
all: seq multicore gpu

seq:
	make -C ./sequential

multicore:
	make -C ./parallel

gpu:
	make -C ./gpu

clean: 
	make clean -C ./sequential
	make clean -C ./parallel
	make clean -C ./gpu

