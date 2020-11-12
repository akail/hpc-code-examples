#!/usr/bin/env pytho3

import math
import sys
import random
import time

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
ranks = comm.Get_size()

master = rank == 0

if master:
    start_time = time.time()
    print("I'm the master process. I control the other process")
else:
    print(f"I'm a computer process {rank}")

sys.stdout.flush()
comm.Barrier()


# Partition the computation out

ITERATIONS = int(sys.argv[1])

def partition(iterations, ranks):
    compute_ranks = ranks-1
    for i in range(1, ranks):
        yield int(iterations/compute_ranks), i


if master:
    for iterations, process in partition(ITERATIONS, ranks):
        request = comm.isend(iterations, dest=process, tag=11)
        request.wait()
    my_iterations = 0
else:
    request = comm.irecv(source=0, tag=11)
    my_iterations = request.wait()

print(f"Process {rank} is computing {my_iterations} points")
sys.stdout.flush()
comm.Barrier()


# Calculate pi
random.seed(2)

def inside_circle():
    x = random.uniform(0,1)
    y = random.uniform(0,1)

    if math.sqrt(x**2 + y**2) <= 1:
        return True
    return False

in_circle = 0
for i in range(my_iterations):
    if inside_circle():
        in_circle += 1

in_circle_points = comm.gather(in_circle, root=0)
if master:
    print("Pi is:", sum(in_circle_points)/ITERATIONS*4)
    end_time = time.time()
    print(f"Calculated in {end_time-start_time} seconds")

