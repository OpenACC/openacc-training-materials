CC=pgcc
CFLAGS=-fast -ta=tesla:pinned -Minfo=accel
CXX=pgc++
CXXFLAGS=-std=c++11 -fast -ta=tesla:cc35,pinned -Minfo=accel
FC=pgfortran
FCFLAGS=-fast -ta=tesla:pinned -Minfo=accel
OPENCV_INCLUDE = -I/usa/efwright/opencv_build/include/opencv4/
OPENCV_LIB     = -L/usa/efwright/opencv_build/lib/ -lopencv_highgui -lopencv_core -lopencv_imgproc

all: run

run: filter.x
	./filter.x ../costarica.jpg out.jpg

filter.x: imageWrapper.o filter.o main.o
	$(CXX) $(CXXFLAGS) -pgf90libs $(OPENCV_INCLUDE) $^ $(OPENCV_LIB) -lm -o $@

.SUFFIXES: .o .cpp .c .F90
.cpp.o: 
	$(CXX) $(CXXFLAGS) $(OPENCV_INCLUDE) $(OPENCV_LIB) -c -o $@ $<
.c.o: 
	$(CC) $(CFLAGS) -c -o $@ $<
.F90.o:
	$(FC) $(FCFLAGS) -c -o $@ $<
.PHONY: clean
clean: 
	-rm -f *.o *.mod core filter.x out.jpg
