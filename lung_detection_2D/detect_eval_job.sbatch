#!/bin/bash
#SBATCH --partition=gpu_min24gb_ext                # Partition (check with "$sinfo")
#SBATCH --output=1-output.out            # Filename with STDOUT. You can use special flags, such as %N and %j.
#SBATCH --error=1-error.out           # (Optional) Filename with STDERR. If ommited, use STDOUT.
#SBATCH --job-name=u_b                 # (Optional) Job name
#SBATCH --qos=gpu_min24gb_ext  

set -euo pipefail

source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate lung_detection_2D

cd /nas-ctm01/homes/jboutet/lung_detection_2D
mkdir -p logs detections_out

INPUT_PATH=$(python - <<'PY'
import pandas as pd
split = 'test'
df = pd.read_csv('/nas-ctm01/homes/jboutet/preprocessing/LIDC-detection/test.csv')
row = df.sample(n=1, random_state=None).iloc[0]
print(row['image_path'])
PY
)
echo "Selected input: $INPUT_PATH"

python detect.py --weights checkpoints/best.pt --input "$INPUT_PATH" --out detections_out
python evaluate_detections.py --config configs/default.yaml --pred-csv detections_out/detections.csv --iou 0.5
