#!/bin/bash

#SBATCH -p dgx2
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --gres gpu:1
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.err
module load miniconda3
source activate av

# training
python train.py --config_file configs/baseline.yaml

# evaluation
python evaluate.py --experiment_path experiments/baseline
