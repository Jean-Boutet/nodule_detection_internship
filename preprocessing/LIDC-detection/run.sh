#!/bin/bash
#SBATCH --partition=cpu_8cores_ext              # Partition (check with "$sinfo")
#SBATCH --qos=cpu_8cores_ext
#SBATCH --output=1-output.out            # Filename with STDOUT. You can use special flags, such as %N and %j.
#SBATCH --error=1-error.out           # (Optional) Filename with STDERR. If ommited, use STDOUT.
#SBATCH --job-name=u_b                 # (Optional) Job name

# Activate the preprocess environment
source /nas-ctm01/homes/jboutet/.conda/envs/preprocess/etc/profile.d/conda.sh
conda activate preprocess

# Commands / scripts to run (e.g., python3 train.py)

python3 preprocess.py




