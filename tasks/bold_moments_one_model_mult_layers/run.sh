#!/usr/bin/env bash
#SBATCH --partition=2080-preemptable-galvani,bethge,2080-galvani
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00

set -euo pipefail

scontrol show job "$SLURM_JOB_ID"

if [ -z "$CONFIG_PATH" ]; then
    echo "Error: Config path is not provided."
    exit 1
fi

srun singularity exec --nv \
    --home "$SCRATCH/home" \
    --pwd "$(pwd)" \
    --bind $SCRATCH \
    --bind /mnt/lustre/work/bethge/ \
    --env PYTHONPATH=multitasking \
    container.sif \
        python -m multitasking.benchmark \
            --config $CONFIG_PATH \
            --output-dir output \
            --overwrite \
            --overwrite-cache
