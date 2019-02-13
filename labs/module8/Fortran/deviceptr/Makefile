F90=pgfortran
F90FLAGS= -ta=tesla:cc35 -Minfo=accel
MCUDA = -Mcuda=cc35 -Minfo=accel -ta=tesla:cc35

main: dot.f90 dot_acc.f90 main.cuf
	$(F90) $(MCUDA) -o $@ $^ 

dot.o:
	$(F90) $(F90FLAGS) -c dot.f90

dot_acc.o:
	$(F90) $(F90FLAGS) -c dot_acc.f90

run:
	./main

clean:
	rm -rf *.o main *.mod
