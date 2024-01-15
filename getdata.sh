#!/bin/bash

#SBATCH --job-name=getdata

#SBATCH --output=./logfiles/getdata.out

#SBATCH --error=./logfiles/getdata.err

#SBATCH --partition=cpu_p1

#SBATCH --time=60:00:00

srun python get_data.py