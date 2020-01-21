#ifndef MODULE7_IMAGE
#define MODULE7_IMAGE

#include <stdio.h>
#include <stdlib.h>
//#include <opencv/highgui.h>

extern "C" unsigned char * readImage(const char*, long&, long&, long&);
extern "C" void writeImage(const char*);

#endif
