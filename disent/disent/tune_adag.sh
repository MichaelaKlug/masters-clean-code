#!/bin/bash
#SBATCH --job-name=tune_adag              # Job name
#SBATCH --partition=bigbatch                     # Replace with your cluster's GPU partition name

# Load your environment
source ~/.bashrc
conda activate mini

# Run your command
python3 adag_optuna.py

