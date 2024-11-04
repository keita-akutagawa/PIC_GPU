#!/bin/bash
#------ pjsub option --------#
#PJM -L rscgrp=share-debug
#PJM -L gpu=1
#PJM --mpi proc=1
#PJM -L elapse=0:10:00
#PJM -g gh76
#PJM -j

#------- Program execution -------#
module load cuda
module load gcc
module load ompi

mpirun -n 1 ./program
