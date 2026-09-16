#!/bin/bash
#SBATCH --job-name=lung_3d_context
#SBATCH --output=logs/train_3d_context_%j.out
#SBATCH --error=logs/train_3d_context_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=cpu_8cores_ext
#SBATCH --qos=cpu_8cores_ext
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -euo pipefail

cd /nas-ctm01/homes/jboutet/lung_detection_3D_one_slide
mkdir -p logs

source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate lung_detection_2D

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

python train_test.py --mode train --config configs/default.yaml --epochs 5 --batch_size 2 --lr 1e-4 --num_workers 0
