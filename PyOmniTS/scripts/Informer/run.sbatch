#!/bin/bash
ml miniconda3
ml cuda/11.8

source ~/.bashrc


# Activate a specific conda environment 
conda activate $smamba_env

# python run.py
# Create a project_dir environment variable in the ~/.bashrc file without the trailing '/' (project_dir=/mnt/home/khanalni/shield-project/extra/S-D-Mamba)
bash /mnt/home/khaledm/PyOmniTS/scripts/Informer/MIMIC_III.sh

