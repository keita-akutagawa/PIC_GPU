#!/bin/bash
#SBATCH --partition=ga40-4gpu
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --time=10:00:00
#SBATCH --output=%x.o%j
#SBATCH --error=%x.e%j

module load nvhpc

export OMPI_MCA_orte_base_help_aggregate=0
export OMPI_MCA_opal_cuda_support=0

mpiexec -n 4 ./program

