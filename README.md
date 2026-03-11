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

The `tasks/benchmark` directory provides an example setup for running the evaluation for
all models on a slurm cluster.

```bash
python tasks/benchmark/launch.py
```


### Leave-one-out inter-subject comparison

Intersubject jobs are started via a separate `launch.py` (and they have their own `run.sh`), but they can use the same config as the main runs (typically `benchmark.yaml`), where you can choose the metrics to run. 

Example command:
```bash
python tasks/bold_moments_benchmark_intersubject_consistency/launch.py \
  --config configs/benchmark.yaml  \
  --output-path outputs/for/your/intersubject/results 
```

### Pairwise inter-subject comparison
Pass an additional parameter `--pairwise` to compare individual subjects' fMRI data. 
Parameter `--n-partners-pairwise` determines how many subjects each target subject is compared to. 
Note: Values above 5 and below 9 will lead to an error, since the current code tries to only create
unique pairs, and with 6+ pairs per subject that is not possible.



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
