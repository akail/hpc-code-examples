
default: build

build:
	mpicc mpitest.c -o mpitest

run: build
	mpirun -n 4 ./mpitest

slurm: build
	sbatch slurm.batch

clean:
	rm mpitest
	rm *.out


	
