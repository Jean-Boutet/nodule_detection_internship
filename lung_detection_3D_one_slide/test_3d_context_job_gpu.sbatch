#!/bin/bash
#SBATCH --partition=gpu_min24gb_ext
#SBATCH --output=logs/test_3d_context_%j.out
#SBATCH --error=logs/test_3d_context_%j.err
#SBATCH --job-name=3d_context_test
#SBATCH --qos=gpu_min24gb_ext

set -euo pipefail

source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate lung_detection_2D

cd /nas-ctm01/homes/jboutet/lung_detection_3D_one_slide
mkdir -p logs

python train_test.py --mode test --config configs/default.yaml --weights checkpoints/3d_context_best.pt --batch_size 2 --num_workers 0
