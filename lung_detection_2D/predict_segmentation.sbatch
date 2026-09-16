#!/bin/bash
#SBATCH --partition=gpu_min12gb_ext                # Partition (check with "$sinfo")
#SBATCH --output=predict_segmentation-%j.out       # STDOUT log
#SBATCH --error=predict_segmentation-%j.err        # STDERR log
#SBATCH --job-name=predict_segmentation            # Job name
#SBATCH --qos=gpu_min12gb_ext


set -euo pipefail

source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate lung_detection_2D

cd /nas-ctm01/homes/jboutet/lung_detection_2D

python predict_segmentation.py \
  --weights checkpoints/segmentation/best.pt \
  --input-npy "/nas-ctm01/homes/jboutet/preprocessing/LIDC-detection/images/LIDC-IDRI-0001_slice_90_img.npy" \
  --mask-path "/nas-ctm01/homes/jboutet/preprocessing/LIDC-detection/masks/LIDC-IDRI-0001_slice_90_mask.npy" \
  --output-image "/nas-ctm01/homes/jboutet/lung_detection_2D/prediction.png" \
  --device cuda
