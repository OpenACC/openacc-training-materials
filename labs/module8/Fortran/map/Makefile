F90=pgfortran
F90FLAGS=-ta=tesla:cc35 -Minfo=accel -Minfo
MCUDA = -Mcuda=cc35 -Minfo -ta=tesla:cc35

#main: dot.f90 dot_acc.f90 main.f90
main: dot.f90 dot_acc.f90 main.cuf
	$(F90) $(F90FLAGS) -o $@ $^  

dot_cuda.o:
	$(F90) $(MCUDA) -c dot_cuda.cuf

dot.o:
	$(F90) -O3 -c dot.f90

run:
	./main

clean:
	rm -rf *.o main *.mod
