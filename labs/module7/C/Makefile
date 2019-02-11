CC=pgcc
CFLAGS=-fast -ta=tesla:cc60,pinned -Minfo=accel
CXX=pgc++
CXXFLAGS=-fast -ta=tesla:cc60,pinned -Minfo=accel

all: filter.x

filter.x: imageWrapper.o filter.o main.o
	$(CXX) $(CXXFLAGS) $^ -lopencv_highgui -lopencv_core -lm -o $@

.SUFFIXES: .o .cpp .c
.cpp.o: 
	$(CXX) $(CXXFLAGS) -c -o $@ $<
.c.o: 
	$(CC) $(CFLAGS) -c -o $@ $<
.PHONY: clean
clean: 
	-rm -f *.o core filter.x
