#!/bin/bash
#SBATCH --job-name=trainDV3_adagad             # Job name
#SBATCH --partition=biggpu                     # Replace with your cluster's GPU partition name

# Load your environment
source ~/.bashrc
conda activate mini

# Run your command
python3 dreamer.py --configs minigrid --task minigrid_unlock --batch_size 2 --logdir ./logdir/minigrid_adag_loss_1000_adversarial