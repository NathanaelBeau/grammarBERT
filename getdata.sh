#!/bin/bash

#SBATCH --job-name=getdata

#SBATCH --output=./logfiles/getdata.out

#SBATCH --error=./logfiles/getdata.err

#SBATCH --ntasks=4

#SBATCH --cpus-per-task=1

#SBATCH --time=50:00:00

module load anaconda-py3/2020.11
conda activate grammarBERT

srun python get_data.py