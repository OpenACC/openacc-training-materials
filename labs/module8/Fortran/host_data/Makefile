F90=pgfortran
F90FLAGS=-ta=tesla:cc35 -Minfo=accel
MCUDA = -Mcuda=cc35 -Minfo=accel -ta=tesla:cc35

main: dot.f90 dot_cuda.cuf main.f90
	$(F90) $(MCUDA) -o $@ $^  

main.o:
	$(F90) $(F90FLAGS) -c main.f90

dot_cuda.o:
	$(F90) $(MCUDA) -c dot_cuda.cuf

dot.o:
	$(F90) -O3 -c dot.f90

run:
	./main

clean:
	rm -rf *.o main *.mod
