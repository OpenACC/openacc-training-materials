.PHONY: all structured unstructured clean
all: structured unstructured

structured:
	make -C ./structured

unstructured:
	make -C ./unstructured

clean: 
	make clean -C ./structured
	make clean -C ./unstructured

