#!/bin/bash
#SBATCH --partition=ga40-4gpu
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --time=0:10:00
#SBATCH --output=%x.o%j
#SBATCH --error=%x.e%j

module load nvhpc

mpiexec -n 4 ./program

