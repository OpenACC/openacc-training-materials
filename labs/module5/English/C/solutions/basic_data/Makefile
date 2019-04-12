CC=pgcc
CXX=pgc++
F90=pgfortran
CFLAGS=-fast -ta=tesla,managed -Minfo=accel
CXXFLAGS=-fast -ta=tesla,managed -Minfo=accel
F90FLAGS=-fast -ta=tesla,managed -Minfo=accel
EXE=main

all: build
	./$(EXE) 4096 4096

build: laplace2d.o jacobi.o
	$(CC) $(CFLAGS) $^ -o $(EXE)

.SUFFIXES: .o .cpp .c .f90
.cpp.o: 
	$(CXX) $(CXXFLAGS) -c -o $@ $<
.c.o: 
	$(CC) $(CFLAGS) -c -o $@ $<
.f90.o:
	$(F90) $(F90FLAGS) -c -o $@ $<
.PHONY: clean
clean: 
	-rm -f *.o core $(EXE)

