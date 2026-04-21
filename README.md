# Alignment Pattern Analysis

This repository provides the code for the ICLR 2026 paper [Only Brains Align with Brains: Cross-Region Alignment Patterns Expose Limits of Normative Models](https://openreview.net/forum?id=cMGJcHHI7d).


## Preliminaries
Setup a virtual environment and install the python package provided by this repository.
Additionally to the dependencies listed in [pyproject.toml](pyproject.toml), you will
need to install the `mmcv` and `mmaction2` for your GPU type, for example:

```bash
# Building MMCV may take some time. Be patient.
# This has to be done *after* torch and torchvision are installed. Otherwise, this
# will install fine but complain about missing `mmcv._ext` later.
MMCV_WITH_OPS=1 FORCE_CUDA="1" TORCH_CUDA_ARCH_LIST="Volta;Turing;Ampere" \
    pip install git+https://github.com/open-mmlab/mmcv.git@v2.1.0

# We additionally need the base config files, so we clone the repository explicitely
# and install the package from there.
# Some source files related to DRN are not copied to site-packages which results in
# an error. To fix this, we use an editable install.
# See https://github.com/open-mmlab/mmaction2/issues/2714
git clone https://github.com/open-mmlab/mmaction2.git
cd mmaction2
git checkout v1.2.0
pip install -e .
```

We provide a definition file for a singularity container that can be built using the
command below. The [container.def](singularity/container.def) might be useful even when
not using singularity, as it defines the exact steps used to set up the environment.

```bash
singularity build --fakeroot singularity/container.sif singularity/container.def
```

Download the BOLDMoments dataset from https://github.com/blahner/BOLDMomentsDataset to the `datasets` directory. Specifically, use the download script for versionB/fsLR32k. Also download the stimuli as detailed at this repository.


## Reproducing alignment pattern analysis

We provide precomputed results from running the benchmarking pipeline. Alignment pattern analysis can be performed based on these results by executing 
```bash
./run_aggregate_results.sh
```
Run figure shell scripts (```run_figure_{x}.sh```) to recreate the paper figures.

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

This repository includes the code for the [Opt-CWM model](https://github.com/neuroailab/Opt_CWM) from [Stojanov et al. (2025)](https://arxiv.org/abs/2503.19953). Please cite all models and datasets used in your work.
