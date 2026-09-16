#!/bin/bash
#SBATCH --partition=gpu_min24gb_ext                # Partition (check with "$sinfo")
#SBATCH --output=logs/test_%j.out
#SBATCH --error=logs/test_%j.err          # (Optional) Filename with STDERR. If ommited, use STDOUT.
#SBATCH --job-name=u_b                 # (Optional) Job name
#SBATCH --qos=gpu_min24gb_ext  

set -euo pipefail

source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate lung_detection_2D

cd /nas-ctm01/homes/jboutet/lung_detection_2D
mkdir -p logs
python train_test.py --mode test --config configs/default.yaml --weights checkpoints/best.pt
