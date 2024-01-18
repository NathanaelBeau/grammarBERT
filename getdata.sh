#!/bin/bash

#SBATCH --job-name=getdata

#SBATCH --qos=qos_gpu-t4

#SBATCH --output=./logfiles/getdata.out

#SBATCH --error=./logfiles/getdata.err

#SBATCH --time=50:30:00

#SBATCH --ntasks=4

#SBATCH --gres=gpu:1

#SBATCH --cpus-per-task=10

#SBATCH --hint=nomultithread

#SBATCH --constraint=v100-32g

module purge
module load anaconda-py3/2019.03
conda activate grammarBERT

set -x
nvidia-smi

srun python get_data.py

