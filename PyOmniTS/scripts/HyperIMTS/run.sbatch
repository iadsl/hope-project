#!/bin/bash

#SBATCH --time=24:00:00              

#SBATCH --ntasks=1             

#SBATCH --cpus-per-task=4

#SBATCH --mem=20G           

#SBATCH --partition=gpu

#SBATCH --gpus-per-node=1

ml miniconda3
ml cuda/11.8

source ~/.bashrc


# Activate a specific conda environment 
conda activate $smamba_env

# python run.py
# Create a project_dir environment variable in the ~/.bashrc file without the trailing '/' (project_dir=/mnt/home/khanalni/shield-project/extra/S-D-Mamba)
bash /mnt/home/khaledm/PyOmniTS/scripts/HyperIMTS/MIMIC_III.sh

