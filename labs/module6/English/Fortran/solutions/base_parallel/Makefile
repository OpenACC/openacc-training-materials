.PHONY: all kernels parallel clean
all: kernels parallel

kernels:
	make -C ./kernels

parallel:
	make -C ./parallel

clean: 
	make clean -C ./kernels
	make clean -C ./parallel

