.PHONY: all seq advanced_data basic_data clean
all: seq advanced_data basic_data

seq:
	make -C ./sequential

advanced_data:
	make -C ./advanced_data

basic_data:
	make -C ./basic_data

clean: 
	make clean -C ./sequential
	make clean -C ./advanced_data
	make clean -C ./basic_data

