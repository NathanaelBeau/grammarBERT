#!/bin/bash

#SBATCH --job-name=getdata

#SBATCH --output=./logfiles/getdata.out

#SBATCH --error=./logfiles/getdata.err

#SBATCH --partition=prepost

#SBATCH --ntasks=4

#SBATCH --cpus-per-task=1

#SBATCH --time=20:00:00

srun python get_data.py