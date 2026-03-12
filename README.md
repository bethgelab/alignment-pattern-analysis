# Alignment Pattern Analysis

This repository provides the code for the ICLR 2026 paper [Only Brains Align with Brains: Cross-Region Alignment Patterns Expose Limits of Normative Models](https://openreview.net/forum?id=cMGJcHHI7d).


## Preliminaries
Setup a virtual environment and install the python package provided by this repository.

We provide a definition file for a singularity container that can be built using the
following command:

```bash
singularity build --fakeroot singularity/container.sif singularity/container.def
```

Download the bold moments dataset to the `datasets` directory.


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
