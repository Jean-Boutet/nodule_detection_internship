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
mkdir -p logs detections_out

read INPUT_PATH MASK_PATH MALIGNANCY <<< $(python - <<'PY'
import pandas as pd
import random

split = random.choice(['train', 'test'])

df = pd.read_csv(
    f'/nas-ctm01/homes/jboutet/preprocessing/LIDC-detection/{split}.csv'
)

row = df.sample(
    n=1,
    random_state=None
).iloc[0]

print(row['image_path'], row['mask_path'], row['malignancy'])
PY
)

echo "Selected input: $INPUT_PATH"
echo "Selected mask : $MASK_PATH"
echo "Selected malignancy: $MALIGNANCY"

python detect2.py \
    --weights checkpoints/best.pt \
    --input "$INPUT_PATH" \
    --mask "$MASK_PATH" \
    --malignancy "$MALIGNANCY" \
    --out detections_out