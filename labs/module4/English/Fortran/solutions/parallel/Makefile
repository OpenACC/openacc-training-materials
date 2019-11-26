CC=pgcc
CXX=pgc++
FC=pgfortran
LD=pgfortran
CFLAGS=-fast -ta=tesla:managed
FFLAGS=-fast -ta=tesla:managed
CXXFLAGS=-fast -ta=tesla:managed
LDFLAGS=-fast -ta=tesla:managed

all: exe run

exe: laplace2d.o jacobi.o
	$(LD) $(LDFLAGS) -o $@ $^

run: exe
	./exe

.PHONY: clean
clean:
	-rm -f *.o *.mod core exe
.SUFFIXES: .c .cpp .F90 .f90 .o
.cpp.o:
	$(CXX) $(CXXFLAGS) -c -o $@ $<
.c.o:
	$(CC) $(CFLAGS) -c -o $@ $<
.F90.o:
	$(FC) $(FFLAGS) -c -o $@ $<
.f90.o:
	$(FC) $(FFLAGS) -c -o $@ $<
