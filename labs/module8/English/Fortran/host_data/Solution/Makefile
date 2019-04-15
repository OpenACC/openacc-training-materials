F90=pgfortran
F90FLAGS=-fast -ta=tesla -Minfo=accel

all: run

run: main
	./main

main: ../dot.f90 ../dot_cuda.cuf main.f90
	$(F90) $(F90FLAGS) -Mcuda $^ -o $@

.PHONY: clean
clean: 
	-rm -f *.o *.mod core main
