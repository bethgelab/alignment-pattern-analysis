#!/usr/bin/env bash
#SBATCH --partition=2080-preemptable-galvani,bethge
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --gres=gpu:0
#SBATCH --time=1-00:00

set -euo pipefail

scontrol show job "$SLURM_JOB_ID"

r3 checkout . "$SCRATCH/job"
cd "$SCRATCH/job"

mkdir -p output

if [ -f "output/done" ]; then
    echo "Job is done already."
    exit 0
fi

run() {
    srun singularity exec \
        --home "$SCRATCH/home" \
        --pwd "$(pwd)/multitasking" \
        --bind $SCRATCH \
        --bind "$SCRATCH:/scratch" \
        --bind $(realpath $R3_REPOSITORY) \
        --bind /mnt/lustre/work/bethge/ \
        --env PYTHONPATH=multitasking \
        container.sif \
            python -m multitasking.render_objaverse \
            --config configs/objaverse.yaml \
            --output-path ../output/dataset
}

if run; then
    echo "completed" > "output/done"
else
    echo "Job failed. Remove the output/done marker to try again."
    echo "failed" > "output/done"
fi
