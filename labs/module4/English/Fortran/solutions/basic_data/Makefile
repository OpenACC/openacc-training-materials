CC=pgcc
CXX=pgc++
FC=pgfortran
LD=pgfortran
CFLAGS=-fast -ta=tesla 
FFLAGS=-fast -ta=tesla 
CXXFLAGS=-fast -ta=tesla 
LDFLAGS=-fast -ta=tesla 

all: exe run

exe: laplace2d.o jacobi.o
	$(LD) $(LDFLAGS) -o $@ $^

run: exe
	./exe

.PHONY: clean
clean:
	-rm -f *.o *.mod core exe
.SUFFIXES: .c .cpp .F90 .o .f90
.cpp.o:
	$(CXX) $(CXXFLAGS) -c -o $@ $<
.c.o:
	$(CC) $(CFLAGS) -c -o $@ $<
.F90.o:
	$(FC) $(FFLAGS) -c -o $@ $<
.f90.o:
	$(FC) $(FFLAGS) -c -o $@ $<
