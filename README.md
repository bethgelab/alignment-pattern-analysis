# multitasking
Taskonomy to brain prediction with a focus on multi-image tasks and the dorsal stream


## Code Release To-Do List

- [x] Remove unused models
- [x] Remove unused metrics
- [ ] Remove unused model api endpoints (demo, FAVORITE_MODELS, default layers?)
- [ ] Clean up analysis scripts
- [ ] Clean up plotting script


## Prerequisites
Build the Singularity container for this project:

```bash
singularity build --fakeroot singularity/multitasking.sif singularity/multitasking.def
```

## Starting the singularity container 
For quick testing. A more comfortable solution would be to use
the container to run your IDE in and directly develop there. 


Don't forget to first start an interactive job:

```bash
srun  --job-name=multitasking --partition=2080-galvani,bethge --cpus-per-task=8 --mem=40000M --gres=gpu:1 --time 72:00:00 --pty bash     
```

We need to mount the data directory (currently the data is on Galvani only),
so a singularity run command could look like this:

```bash 
CONTAINER="singularity/multitasking.sif"

singularity run -p --nv \
    --pwd $(pwd) \
    --bind "/mnt/lustre/work" \
    --bind "/mnt/lustre/datasets" \
    $CONTAINER
```

## Evaluating models

To evaluate a model in terms of similarity to human fMRI responses on the fMRI-Objaverse
dataset, run the following command:

```bash
python -m multitasking.benchmark --config configs/benchmark.yaml
```

To test beforehand whether a model runs on your setup, use the interactive demo script:

```bash
python -m multitasking.models.demo
```

## Running the complete benchmark
To run the complete benchmark, distributed across the Slurm cluster, you will need a
local Python environment with `click`, `executor` and `python-benedict` installed. I
(Matthias) use a conda environment for this. Furthermore, you need to build the
multitasking container as described above.

The benchmark can then be launched with:

```bash
python tasks/benchmark/launch.py --output-path output/benchmark
```

This will prepare jobs for all models defined in `configs/models/`, using 
`configs/benchmark.yaml` as a base config. The benchmarking outputs will be saved to
`output/benchmark/outputs`, and the logs to `output/benchmark/logs`.

Have a look at the `tasks/benchmark/launch.py` script to see further options.

## Intersubject similarities

### Standard leave-one-subject-out method

Intersubject jobs are started via a separate `launch.py` (and they have their own `run.sh`), but they can use the same config as the main runs (typically `benchmark.yaml`), where you can choose the metrics to run. 

Example command:
```bash
python tasks/bold_moments_benchmark_intersubject_consistency/launch.py \
  --config configs/benchmark.yaml  \
  --output-path outputs/for/your/intersubject/results 
```

### Pairwise:
Pass an additional parameter `--pairwise` to compare individual subjects' fMRI data. 
Parameter `--n-partners-pairwise` determines how many subjects each target subject is compared to. 
Note: Values above 5 and below 9 will lead to an error, since the current code tries to only create
unique pairs, and with 6+ pairs per subject that is not possible.


## Loading fMRI data
To load fMRI data:

```bash
python -m fmri_data.load_fmri --config configs/example_with_data.yaml
```

## Plotting results

You can create an overview plot for BOLDMoments benchmarking results using 
```console
python -m multitasking.plot_creation.summary_plots --output-supdir path/to/benchmarking/results --filename scores_rsa.csv --plot-path /path/where/to/save/plots 
```
`--output-supdir`: the directory that contains subdirectories with your results like so: output_supdir/{hash}/scoresheets/scores_rsa.csv

`--plot-path`: directory where you want to save your plots & results files

This defaults to using the test split.

To add a noise ceiling, add an option:
`--intersubject-supdir` pointing to the intersubject results to use as noise ceiling. Note this currently doesn't check for the right subject being used. 


## Contributing
All contributions are welcome! Please open a PR or issue on GitHub.

The codebase comes with some tools to ensure code quality:

```bash
# Make sure you have the dev requirements installed
pip install -r requirements-dev.txt

# Run linter (ruff) and type checker (mypy)
make lint

# Run ruff only
make ruff

# Resolve issues for which automatic fixes are available and auto-format your code
make fix

# Run mypy only
make mypy

# Run the tests
make test
```

When submitting a PR, GitHub Actions will run the tests and linters for you. Please make
sure to fix any issues before submitting.

To have the linters run automatically before you commit, you can install the pre-commit
hooks:

```bash
pre-commit install
```
