.PHONY: all kernels parallel clean
all: kernels parallel

kernels:
	make -C ./base_parallel

parallel:
	make -C ./collapse

clean: 
	make clean -C ./kernels
	make clean -C ./parallel

