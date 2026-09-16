#!/bin/bash
#SBATCH --partition=gpu_min12gb_ext                # Partition (check with "$sinfo")
#SBATCH --output=segmentation-%j.out                # STDOUT log
#SBATCH --error=segmentation-%j.err                 # STDERR log
#SBATCH --job-name=segmentation                     # Job name
#SBATCH --qos=gpu_min12gb_ext


set -euo pipefail

source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate lung_detection_2D

cd /nas-ctm01/homes/jboutet/lung_detection_2D

python train_segmentation.py \
  --train-csv "/nas-ctm01/homes/jboutet/preprocessing/LIDC-detection/train.csv" \
  --val-csv "/nas-ctm01/homes/jboutet/preprocessing/LIDC-detection/test.csv" \
  --images-dir "/nas-ctm01/homes/jboutet/preprocessing/LIDC-detection/images" \
  --masks-dir "/nas-ctm01/homes/jboutet/preprocessing/LIDC-detection/masks" \
  --save-dir checkpoints/segmentation \
  --batch-size 8 \
  --epochs 50 \
  --num-workers 4 \
  --device cuda \
  --amp
