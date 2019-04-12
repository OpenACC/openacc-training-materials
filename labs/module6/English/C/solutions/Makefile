.PHONY: all base_parallel collapse gang_worker_vector reduction sequential tile clean
all: base_parallel collapse gang_worker_vector reduction sequential tile

base_parallel:
	make -C ./base_parallel

collapse:
	make -C ./collapse

gang_worker_vector:
	make -C ./gang_worker_vector

reduction:
	make -C ./reduction

sequential:
	make -C ./sequential

tile:
	make -C ./tile

clean: 
	make clean -C ./sequential
	make clean -C ./collapse
	make clean -C ./gang_worker_vector
	make clean -C ./reduction
	make clean -C ./sequential
	make clean -C ./tile

