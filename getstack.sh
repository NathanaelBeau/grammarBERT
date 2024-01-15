#!/bin/bash

#SBATCH --job-name=get_baselines

#SBATCH --output=./logfiles/download.out

#SBATCH --error=./logfiles/download.err

#SBATCH --partition=prepost

module load git-lfs
git lfs install
git clone https://huggingface.co/datasets/bigcode/the-stack