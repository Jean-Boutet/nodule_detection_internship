#!/bin/bash
#SBATCH --partition=gpu_min24gb_ext                # Partition (check with "$sinfo")
#SBATCH --output=2-output.out            # Filename with STDOUT. You can use special flags, such as %N and %j.
#SBATCH --error=2-error.out           # (Optional) Filename with STDERR. If ommited, use STDOUT.
#SBATCH --job-name=detect                 # (Optional) Job name
#SBATCH --qos=gpu_min24gb_ext  

set -euo pipefail

source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate lung_detection_2D

cd /nas-ctm01/homes/jboutet/lung_detection_2D
python shape.py