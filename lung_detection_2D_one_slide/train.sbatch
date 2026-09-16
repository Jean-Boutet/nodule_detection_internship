#!/bin/bash
#SBATCH --partition=gpu_min24gb_ext                # Partition (check with "$sinfo")
#SBATCH --output=1-output.out            # Filename with STDOUT. You can use special flags, such as %N and %j.
#SBATCH --error=1-error.out           # (Optional) Filename with STDERR. If ommited, use STDOUT.
#SBATCH --job-name=u_b                 # (Optional) Job name
#SBATCH --qos=gpu_min24gb_ext                           # (Optional) 01.ctm-deep-05

# Commands / scripts to run (e.g., python3 train.py)

set -euo pipefail

cd /nas-ctm01/homes/jboutet/lung_detection_2D_one_slide
mkdir -p logs

source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate lung_detection_2D

python train_test.py --mode train --config configs/default.yaml --batch_size 8 --epochs 150 --lr 1e-4 --central_only