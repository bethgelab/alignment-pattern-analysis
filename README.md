# multitasking
Taskonomy to brain prediction with a focus on multi-image tasks and the dorsal stream


## Code Release To-Do List

- [x] Remove unused models
- [x] Remove unused metrics
- [ ] Remove unused model api endpoints (demo, FAVORITE_MODELS, default layers?)
- [ ] Clean up analysis scripts
- [ ] Clean up plotting script
- [ ] Remove all slurm-related code
- [ ] Remove tasks and use benchmark as single entry point
- [ ] Add license


## Installation
Setup a virtual environment and install the python package provided by this repository.
We recommend using [uv](https://docs.astral.sh/uv/), which allows using the precise
dependency version from the [uv.lock](uv.lock) file:

```bash
uv sync
```


## Dataset Preparation
Download the bold moments dataset to the `datasets` directory. **TODO: Add more precise instructions**.



## Evaluating models
**TODO: Revise once there's a clean entry point to the benchmark**

Use the [`multitasking.benchmark`](multitasking/benchmark.py) script to evaluate models:

```bash
# Evaluate all models
python -m multitasking.benchmark

# Evaluate a specific model
python -m multitasking.benchmark --model timm/resnet18.a1_in1k
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



## Visualizing results
**TODO: Revise plotting instruction**

You can create an overview plot for BOLDMoments benchmarking results using 
```console
python -m multitasking.plot_creation.summary_plots --output-supdir path/to/benchmarking/results --filename scores_rsa.csv --plot-path /path/where/to/save/plots 
```
`--output-supdir`: the directory that contains subdirectories with your results like so: output_supdir/{hash}/scoresheets/scores_rsa.csv

`--plot-path`: directory where you want to save your plots & results files

This defaults to using the test split.

To add a noise ceiling, add an option:
`--intersubject-supdir` pointing to the intersubject results to use as noise ceiling. Note this currently doesn't check for the right subject being used. 



## Citation
**TODO: Add bibtex reference and code for our paper**
