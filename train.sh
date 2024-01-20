#!/bin/bash

#SBATCH --job-name=retrocode

#SBATCH --qos=qos_gpu-t4

#SBATCH --output=./logfiles/logfile_grammarbert.out

#SBATCH --error=./logfiles/logfile_grammarbert.err

#SBATCH --time=40:30:00

#SBATCH --ntasks=1

#SBATCH --gres=gpu:1

#SBATCH --cpus-per-task=1

#SBATCH --hint=nomultithread

#SBATCH --constraint=v100-32g


module purge
module load anaconda-py3/2019.03
conda activate grammarBERT
set -x
nvidia-smi
# This will create a config file on your server


srun python finetuningcodebertmlm.py