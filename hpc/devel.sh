#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=1:00
#SBATCH --output=./hpc/slurm/%j

make $1
./bin/$1
