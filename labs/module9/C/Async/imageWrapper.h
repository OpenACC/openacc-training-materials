#ifndef MODULE7_IMAGE
#define MODULE7_IMAGE

#include <stdio.h>
#include <stdlib.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
//#include <opencv/highgui.h>

unsigned char * readImage(const char*, long&, long&, long&);
void writeImage(const char*);

#endif
