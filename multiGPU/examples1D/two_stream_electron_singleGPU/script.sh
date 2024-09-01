#!/bin/bash
#SBATCH --partition=ga40-1gpu
#SBATCH --gres=gpu:1
#SBATCH --time=120:00:00
#SBATCH --output=%x.o%j
#SBATCH --error=%x.e%j

module load nvhpc

./program

