F90=pgfortran
F90FLAGS=-Mcuda=cc35 -ta=tesla:cc35 -Minfo=accel
MCUDA = -Mcuda=cc35 -Minfo=accel -ta=tesla:cc35

all: clean main
	./main

main: dot.o dot_acc.o main.o
	$(F90) $(F90FLAGS) -o $@ $^  

main.o:
	$(F90) $(F90FLAGS) $(MCUDA) -c main.cuf

dot_acc.o:
	$(F90) $(F90FLAGS) -c dot_acc.f90

dot.o:
	$(F90) -c dot.f90

run:
	./main

clean:
	-rm -rf *.o main *.mod
