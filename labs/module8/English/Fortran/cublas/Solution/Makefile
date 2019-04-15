F90=pgfortran
F90FLAGS=-fast -ta=tesla -Minfo=accel

all: run

run: main
	./main

main: ../matmul.f90 ../matmul_cublas.f90 main.f90
	$(F90) $(F90FLAGS) -Mcuda -lcublas $^ -o $@

.PHONY: clean
clean: 
	-rm -f *.o *.mod core main
