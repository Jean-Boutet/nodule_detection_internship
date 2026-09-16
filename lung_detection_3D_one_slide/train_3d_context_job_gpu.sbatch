#!/bin/bash
#SBATCH --partition=gpu_min24gb_ext
#SBATCH --output=logs/train_3d_context_%j.out
#SBATCH --error=logs/train_3d_context_%j.err
#SBATCH --job-name=3d_context_train
#SBATCH --qos=gpu_min24gb_ext

set -euo pipefail

source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate lung_detection_2D

cd /nas-ctm01/homes/jboutet/lung_detection_3D_one_slide
mkdir -p logs

python train_test.py --mode train --config configs/default.yaml --epochs 150 --batch_size 2 --lr 1e-4 --num_workers 0
