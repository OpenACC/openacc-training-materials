F90=pgfortran
F90FLAGS=-ta=tesla:cc35 -Minfo=accel -Minfo -lcublas
MCUDA = -Mcuda=cc35 -Minfo -ta=tesla:cc35 -lcublas

all: clean main
	./main

main: ../matmul.f90 ../matmul_cublas.f90 main.f90
	$(F90) $(MCUDA) -o $@ $^ 

run:
	./main

clean:
	-rm -rf *.o main *.mod

