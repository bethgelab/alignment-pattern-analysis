#!/usr/bin/env bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=0-08:00
#
# No --gres: inter-subject consistency compares brain to brain. It builds the
# dataset and scores with numpy/sklearn and never instantiates a model, so it
# asks for no GPU and runs singularity without --nv. Point it at a CPU
# partition (see "Cluster configuration" in the README) and it stops idling on
# a GPU and queueing behind the model benchmark for one.
#
# The directives above are portable defaults. Nothing here names a cluster:
# the partition, the account and any other site-specific sbatch flags are read
# from the environment by launch.py and passed on the sbatch command line,
# which overrides the matching directive above. sbatch parses these lines as
# comments before any shell runs, so they cannot expand variables themselves.

set -euo pipefail

scontrol show job "$SLURM_JOB_ID"

if [ -z "${CONFIG_PATH:-}" ]; then
    echo "Error: Config path is not provided."
    exit 1
fi

if [ -z "${CONTAINER:-}" ]; then
    echo "Error: Container is not provided."
    exit 1
fi

if [ -z "${SCRATCH:-}" ]; then
    echo "Error: SCRATCH is not set. It must point at a writable directory."
    exit 1
fi

# Host paths the container cannot see by default -- typically the filesystem
# holding the datasets and the output directory. Comma-separated, may be empty.
bind_args=()
if [ -n "${MULTITASKING_BIND_PATHS:-}" ]; then
    IFS=',' read -ra bind_paths <<< "$MULTITASKING_BIND_PATHS"
    for bind_path in "${bind_paths[@]}"; do
        if [ -n "$bind_path" ]; then
            bind_args+=(--bind "$bind_path")
        fi
    done
fi

mkdir -p "$SCRATCH/home"

srun singularity exec \
    --home "$SCRATCH/home" \
    --pwd "$(pwd)" \
    --bind "$SCRATCH" \
    ${bind_args[@]+"${bind_args[@]}"} \
    --env PYTHONPATH=multitasking \
    "$CONTAINER" \
        python -m multitasking.intersubject_consistency \
            --config "$CONFIG_PATH" \
            --output-dir output \
            --overwrite
