#!/bin/bash
#SBATCH --job-name=tune_dv2dflat             # Job name
#SBATCH --partition=bigbatch                   # Replace with your cluster's GPU partition name

# Load your environment
source ~/.bashrc
conda activate mini

# Run your command
python3 adag_dv3_optuna.py

