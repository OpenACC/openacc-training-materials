CC=pgcc
CXX=pgc++
FC=pgfortran
LD=pgcc
CFLAGS=-fast -ta=multicore 
FFLAGS=-fast -ta=multicore 
CXXFLAGS=-fast -ta=multicore 
LDFLAGS=-fast -ta=multicore 

all: exe run

exe: laplace2d.o jacobi.o
	$(LD) $(LDFLAGS) -o $@ $^

run: exe
	./exe

.PHONY: clean
clean:
	-rm -f *.o *.mod core exe
.SUFFIXES: .c .cpp .F90 .o
.cpp.o:
	$(CXX) $(CXXFLAGS) -c -o $@ $<
.c.o:
	$(CC) $(CFLAGS) -c -o $@ $<
.F90.o:
	$(FC) $(FFLAGS) -c -o $@ $<
