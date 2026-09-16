#!/bin/bash
#SBATCH --job-name=lung_detect
#SBATCH --output=logs/detect_%j.out
#SBATCH --error=logs/detect_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=cpu_8cores_ext
#SBATCH --qos=cpu_8cores_ext
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -euo pipefail

source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate lung_detection_2D

cd /nas-ctm01/homes/jboutet/lung_detection_2D
mkdir -p logs

INPUT_PATH=$(python - <<'PY'
import pandas as pd
import random
split = random.choice(['train', 'test'])
df = pd.read_csv(f'/nas-ctm01/homes/jboutet/preprocessing/LIDC-detection/{split}.csv')
row = df.sample(n=1, random_state=None).iloc[0]
print(row['image_path'])
PY
)
echo "Selected input: $INPUT_PATH"
python detect.py --weights checkpoints/best.pt --input "$INPUT_PATH" --out detections_out
