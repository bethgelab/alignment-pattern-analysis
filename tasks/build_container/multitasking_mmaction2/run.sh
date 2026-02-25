#!/usr/bin/env bash
#SBATCH --partition=2080-preemptable-galvani,bethge
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --gres=gpu:0
#SBATCH --time=0-08:00

set -euo pipefail

scontrol show job "$SLURM_JOB_ID"

SIF_FILE=output/container.sif
DEF_FILE=multitasking/singularity/multitasking_mmaction2.def

export APPTAINER_CACHEDIR=${SCRATCH:-/scratch_local/bethge/$USER}
export APPTAINER_TMPDIR=${SCRATCH:-/scratch_local/bethge/$USER}
export SINGULARITY_CACHEDIR=$APPTAINER_CACHEDIR
export SINGULARITY_TMPDIR=$APPTAINER_TMPDIR

r3 checkout . "$SCRATCH/job"
cd "$SCRATCH/job"

mkdir -p output

if [ -f "output/done" ]; then
    echo "Job is done already."
    exit 0
fi

if srun singularity build --fakeroot "$SIF_FILE" "$DEF_FILE"; then
    echo "completed" > "output/done"
else
    echo "Job failed. Remove the output/done marker to try again."
    echo "failed" > "output/done"
fi
