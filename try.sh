#!/bin/bash

#SBATCH --job-name=grammarbert

#SBATCH --qos=qos_gpu-dev

#SBATCH --output=./logfiles/test.out

#SBATCH --error=./logfiles/test.err

#SBATCH --time=00:30:00

#SBATCH --ntasks=1

#SBATCH --gres=gpu:4

#SBATCH --cpus-per-task=10

#SBATCH --hint=nomultithread

#SBATCH --constraint=v100-32g


module purge
module load anaconda-py3/2019.03
conda activate grammarBERT
set -x
nvidia-smi
# This will create a config file on your server


srun python train_streaming.py