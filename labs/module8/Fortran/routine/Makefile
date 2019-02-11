F90=pgfortran
F90FLAGS= -ta=tesla:cc35 -Minfo=accel -Minfo
MCUDA = -Mcuda=cc35 -Minfo -ta=tesla:cc35

main: distance_map.f90 dist_cuda.cuf distance_map_acc.f90 main.f90
	$(F90) $(MCUDA) -o $@ $^ 

run:
	./main

clean:
	rm -rf *.o main *.mod
