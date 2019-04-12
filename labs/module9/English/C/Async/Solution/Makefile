CC=pgcc
CFLAGS=-fast -ta=tesla:cc35,pinned -Minfo=accel
CXX=pgc++
CXXFLAGS=-std=c++11 -fast -ta=tesla:cc35,pinned -Minfo=accel
OPENCV_INCLUDE = -I/usa/efwright/opencv_build/include/opencv4/
OPENCV_LIB     = -L/usa/efwright/opencv_build/lib/ -lopencv_highgui -lopencv_core -lopencv_imgproc

all: run

run: filter.x
	./filter.x ../costarica.jpg out.jpg

filter.x: imageWrapper.o filter.o main.o
	$(CXX) $(CXXFLAGS) $(OPENCV_INCLUDE) $^ $(OPENCV_LIB) -lm -o $@

.SUFFIXES: .o .cpp .c
.cpp.o: 
	$(CXX) $(CXXFLAGS) $(OPENCV_INCLUDE) $(OPENCV_LIB) -c -o $@ $<
.c.o: 
	$(CC) $(CFLAGS) -c -o $@ $<
.PHONY: clean
clean: 
	-rm -f *.o core filter.x out.jpg
