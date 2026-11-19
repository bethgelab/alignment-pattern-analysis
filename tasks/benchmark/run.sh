#!/usr/bin/env bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --time=0-08:00
#
# The directives above are portable defaults.
# See "Cluster configuration" in the README for the variables involved.

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

# Optional persistent torch.hub cache (hub repos and checkpoints). Without it the
# cache lives in the container home under $SCRATCH, which is node-local on most
# clusters, so every job re-downloads the weights (5-16 GB per V-JEPA 2 model).
# The directory must be visible in the container, e.g. via MULTITASKING_BIND_PATHS.
env_args=()
if [ -n "${MULTITASKING_TORCH_HOME:-}" ]; then
    mkdir -p "$MULTITASKING_TORCH_HOME"
    env_args+=(--env "TORCH_HOME=$MULTITASKING_TORCH_HOME")
fi

container_args=(
    --home "$SCRATCH/home"
    --pwd "$(pwd)"
    --bind "$SCRATCH"
    ${bind_args[@]+"${bind_args[@]}"}
)

# Jobs must run the copy of multitasking/ in this job directory, never an
# installed one: the image carries no copy, and this fails the job if one appears.
singularity exec "${container_args[@]}" "$CONTAINER" python -c '
import os, sys, multitasking
found = os.path.realpath(multitasking.__file__)
if not found.startswith(os.path.realpath("multitasking") + os.sep):
    sys.exit(f"multitasking imported from {found}, not from this job directory")
'

srun singularity exec --nv "${container_args[@]}" \
    ${env_args[@]+"${env_args[@]}"} \
    "$CONTAINER" \
        python -m multitasking.benchmark \
            --config "$CONFIG_PATH" \
            --output-dir output \
            --overwrite \
            --overwrite-cache
