# Alignment Pattern Analysis

This repository provides the code for the ICLR 2026 paper [Only Brains Align with Brains: Cross-Region Alignment Patterns Expose Limits of Normative Models](https://openreview.net/forum?id=cMGJcHHI7d).


## Preliminaries
Setup a virtual environment and install the python package provided by this repository.

We provide a definition file for a singularity container that can be built using the
following command:

```bash
singularity build --fakeroot singularity/container.sif singularity/container.def
```

Download the BOLDMoments dataset from https://github.com/blahner/BOLDMomentsDataset to the `datasets` directory. Specifically, use the download script for versionB/fsLR32k. Also download the stimuli as detailed at this repository.


## Evaluating models

Use the [`multitasking.benchmark`](multitasking/benchmark.py) script to evaluate a
single model:

```bash
python -m multitasking.benchmark --config configs/benchmark.yaml
```

The `tasks/` directory provides example setups for running the evaluation for multiple
models or subjects in parallel on a slurm cluster.

```bash
# Run all models
python tasks/benchmark/launch.py --output-path path/to/output

# Run leave-one-out inter-subject comparison for all subjects
python tasks/benchmark_inter_subject_consistency/launch.py --output-path path/to/output

# Run pairwise inter-subject comparison for all subjects
python tasks/benchmark_inter_subject_consistency/launch.py --pairwise --output-path path/to/output
```


## Cluster configuration

The scripts in `tasks/` name no cluster. Everything site-specific -- which
partitions exist, which account to bill, where the datasets live -- is read
from the environment, so the same scripts run anywhere with SLURM and
singularity. Set the variables below before calling a launcher; none of them
is required, and with none set the jobs fall back to your cluster's defaults
and to the portable `#SBATCH` directives in the `run.sh` files.

| Variable | Used by | Meaning |
| --- | --- | --- |
| `MULTITASKING_SLURM_PARTITION` | both launchers | Comma-separated partitions for the model benchmark, which needs a GPU. |
| `MULTITASKING_SLURM_CPU_PARTITION` | inter-subject launcher | Comma-separated partitions for the inter-subject jobs, which need no GPU. Falls back to `MULTITASKING_SLURM_PARTITION` when unset. |
| `MULTITASKING_SLURM_ACCOUNT` | both launchers | Account to bill, passed as `--account`. |
| `MULTITASKING_SLURM_EXTRA_ARGS` | both launchers | Any further `sbatch` flags, passed through verbatim, e.g. `--qos high --constraint a100`. |
| `MULTITASKING_BIND_PATHS` | both `run.sh` | Comma-separated host paths to bind into the container, typically the filesystem holding the datasets. `$SCRATCH` is always bound and need not be listed. |
| `SCRATCH` | both `run.sh` | A writable directory for the container home and for intermediate results. Must be set. |

For example:

```bash
export SCRATCH=/path/to/scratch
export MULTITASKING_BIND_PATHS=/path/to/datasets,/path/to/results
export MULTITASKING_SLURM_PARTITION=gpu-short,gpu-long
export MULTITASKING_SLURM_CPU_PARTITION=cpu
export MULTITASKING_SLURM_ACCOUNT=my-account

python tasks/benchmark/launch.py --output-path path/to/output
```

Use `--dry-run` to print the `sbatch` command lines without submitting
anything, which is the quickest way to check that your settings come through
as you expect.

Two notes on how this works. The partition and account arrive as `sbatch`
command-line flags rather than as `#SBATCH` lines, because `sbatch` reads
those lines as comments before any shell runs and so cannot expand variables
in them; a command-line flag overrides the matching directive in the script.
And a config file may still carry its own `slurm:` block, which is applied
after the environment and therefore wins over it for that one run.


## Citation
```bibtex
@inproceedings{
  title={Only Brains Align with Brains: Cross-Region Alignment Patterns Expose Limits of Normative Models},
  author={H{\"o}fling, Larissa and Tangemann, Matthias and Piefke, Lotta and Keller, Susanne and Bethge, Matthias and Franke, Katrin},
  booktitle={ICLR},
  year={2026},
  url={https://openreview.net/forum?id=cMGJcHHI7d}
}
```

This repository includes the code for the [Opt-CWM model](https://github.com/neuroailab/Opt_CWM) from [Stojanov et al. (2025)](https://arxiv.org/abs/2503.19953). Please cite all models and datasets used in your work.
