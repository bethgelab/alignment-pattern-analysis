#!/usr/bin/env bash
# Inter-subject consistency compares brain to brain: it builds the dataset
# and scores with numpy/sklearn, never instantiating a model. So it asks for
# no GPU and can also run on the CPU partition, instead of idling on a GPU
# and queueing behind the model benchmark for one.
#SBATCH --partition=cpu-galvani,2080-preemptable-galvani,bethge,2080-galvani
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=0-08:00

set -euo pipefail

scontrol show job "$SLURM_JOB_ID"

if [ -z "$CONFIG_PATH" ]; then
    echo "Error: Config path is not provided."
    exit 1
fi

if [ -z "$CONTAINER" ]; then
    echo "Error: Container is not provided."
    exit 1
fi

mkdir -p "$SCRATCH/home"

srun singularity exec \
    --home "$SCRATCH/home" \
    --pwd "$(pwd)" \
    --bind $SCRATCH \
    --bind /mnt/lustre/work/bethge/ \
    --env PYTHONPATH=multitasking \
    $CONTAINER \
        python -m multitasking.intersubject_consistency \
            --config $CONFIG_PATH \
            --output-dir output \
            --overwrite
