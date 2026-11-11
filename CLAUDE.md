# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Code for the ICLR 2026 paper *Only Brains Align with Brains: Cross-Region Alignment
Patterns Expose Limits of Normative Models*. The `multitasking` package evaluates how
well vision-model representations align with fMRI responses, per brain region, and
compares those cross-region alignment patterns against brain-brain (inter-subject)
patterns. Three datasets are supported: BOLD Moments (videos, fsLR32k, `sub-NN`), NSD
(still images, fsaverage, `subjNN`), and NSD-synthetic (220 non-naturalistic still
images, the same eight NSD subjects and fsaverage parcels). NSD-synthetic exists only as
a *test* set: the encoding model is fitted on NSD-core and scored on it, which makes the
alignment patterns an out-of-distribution test.

## Commands

```bash
make lint          # ruff check + mypy
make ruff          # ruff check .
make fix           # ruff check --fix
make mypy          # mypy .
make test          # python -m pytest --cov=multitasking

pytest tests/test_nsd_splits.py                       # one file
pytest tests/test_nsd_splits.py::test_name             # one test
pytest tests/test_dataset_nsd.py -k "not real"         # skip real-data tests
```

Dev deps come from the `dev` dependency group (`uv sync --group dev`, or install into a
venv and run the bare commands as the Makefile does).

Single run, full SLURM sweep, and analysis:

```bash
python -m multitasking.benchmark --config configs/benchmark.yaml
python -m multitasking.benchmark --config configs/benchmark_nsd_to_synthetic.yaml
python tasks/benchmark/launch.py --output-path /some/job/dir
python tasks/benchmark_inter_subject_consistency/launch.py [--pairwise] --output-path /some/job/dir
SCRATCH=... DATASET=nsd OUTPUT_SUPDIR=... INTERSUBJECT_DIR=... PAIRWISE_SUBJECT_DIR=... \
  ./run_aggregate_results.sh
DATASET=nsd ./run_figure_0_all.sh && DATASET=nsd ./run_figure_1.sh
```

Both launchers take `--dry-run` to print the `sbatch` lines without submitting.

## Architecture

### The pipeline

`multitasking/benchmark.py` is the single entrypoint for one (model, dataset) pair. It
is entirely config-driven — a YAML file parsed with `benedict`, passed around as a dict,
and read with dotted keys — and runs:

1. `build_dataset(config, split)` → one dataset object per split (`datasets.py`). When
   the config carries a `test_split` subtree, the test split is built from a *different*
   dataset than the train split; see "Cross-dataset runs" below.
2. `compute_model_representation` (`utils/get_representations.py`) → feature extraction
   (`feature_extraction.py`) then dimensionality reduction (`feature_reduction.py`,
   default sparse random projection). The train split fits the projectors; the test
   split **must** reuse them, so they are threaded through explicitly.
3. `dataset.load_fmri_data()` and `load_voxel_consistency()`, optionally masking voxels
   below `fmri.voxel_consistency_threshold`.
4. `compute_rsa` and `compute_linear_predictivity` per (layer, ROI, split, subject),
   written as CSV scoresheets.

`multitasking/intersubject_consistency.py` is the same pipeline with one subject (or a
leave-one-out mean of subjects) standing in for the model. It imports `compute_rsa` and
`compute_linear_predictivity` from `benchmark.py` unchanged, and encodes the source
subject(s) in `feature_extraction.model`; the other `feature_extraction` keys are
ignored. Its extra config lives under `fmri.intersubject` and is normally written by the
launcher rather than by hand.

### The sample-ordering contract

`multitasking/datasets_base.py` defines `FmriStimulusDataset`, which both `BoldMoments`
(`datasets.py`) and `NSD` (`datasets_nsd.py`) implement. Deliberately a contract, not
shared implementation — the two datasets differ in stimulus medium, surface space,
subject naming and file layout.

The load-bearing part is `sample_ids`: model features are extracted once per split and
reused for every subject, so row `i` of every array from `load_fmri_data()` must
correspond to `sample_ids[i]`, and the ordering must not depend on which subjects are
scored. Anything touching the dataset layer has to preserve this;
`tests/test_dataset_bold_moments_regression.py` pins it by hashing the ordering.

### Config hashing decides where things land

`utils/file_handling.config_hash` takes an 8-char SHA-256 prefix of the config.

- Output goes to `<output-dir>/<hash of the full config>/`, containing `config.yaml`,
  `scoresheets/scores_{rsa,linear_predictivity}_{subject}.csv`, `ridgecv/`, and
  optionally `rdm_heatmaps/`.
- Cache goes to `<cache-dir>/<hash of dataset+feature_extraction+fmri>/`, with the
  feature-reduction cache keyed by a second hash that also includes `feature_reduction`.

So *any* config edit changes the output directory, and results of one sweep are many
hash directories side by side. The analysis layer globs `scoresheets/*.csv` recursively
rather than looking anything up by name. Use `--overwrite-cache` for results you intend
to publish; the cache is a development convenience.

`test_split` is included in both hash key lists, so a cross-dataset run never shares a
feature cache with a plain one. Because `config_hash` filters by key, adding it left
every existing hash unchanged.

### Cross-dataset runs

`configs/benchmark_nsd_to_synthetic.yaml` fits on NSD-core and scores on NSD-synthetic.
The mechanism is a `test_split:` subtree holding `dataset` and `fmri` keys that
`build_dataset` deep-merges when `split == "test"`, returning a new config rather than
mutating the caller's.

Only `datasets.TEST_SPLIT_OVERRIDABLE_KEYS` may be overridden, and that whitelist is
load-bearing. Several settings are read from the *train* dataset or the base config and
applied to both splits: `subject_ids` comes from `dataset_train`, `roi_names` from the
base config, and the voxel-consistency mask is built once from the train dataset's NCSNR
as a positional vector in ROI-vertex order. So the two splits must share an atlas, a
subsample stride, an ROI list *and its order*, and a stimulus resolution.
`benchmark.check_splits_are_comparable` asserts exactly that, because an ROI *ordering*
mismatch would otherwise be silent.

A consequence worth remembering: on the benchmark path the voxel mask always comes from
the train dataset, so in a cross-dataset run it is NSD-core's NCSNR regardless of
`fmri.ncsnr_source`. That setting governs `load_voxel_consistency`, which the
inter-subject path and the noise-ceiling normalisation use.

Scoresheets carry a `stimulus_set` column for the same reason: `split` alone is ambiguous
once "test" can mean either dataset, and `aggregate_results.py` both de-duplicates and
merges noise ceilings on `split`. Normalising a synthetic score by an NSD-core ceiling
would otherwise go through silently.

### `$SCRATCH` is load-bearing

Defaulting to `/scratch`, it holds the feature cache and — more importantly — is the
handoff between `analysis_scripts/aggregate_results.py` (which pickles its dataframes
there) and `plotting_scripts/figure_*.py` (which read them back). Inside a SLURM job
`$SCRATCH` is node-local and deleted when the job ends, so set it explicitly to a
persistent directory when aggregating results you intend to plot.

### Models

`models/build_model("provider/name")` splits on the first `/` and imports
`multitasking.models.<provider>`; each provider module exposes `build_model`,
`list_models`, and its model classes subclass `models/base.py`'s `Model` (or
`ImageModel`, which applies an image model frame-by-frame and reshapes outputs to
`(B, C, T, H, W)`). Adding a provider means adding a module and listing it in
`list_providers()`. Features are captured with forward hooks (`LayerOutputRecorder`),
so config `feature_extraction.layers` entries are `named_modules()` paths.

`configs/models/*.yaml` each override only the `feature_extraction` subtree; the SLURM
launcher deep-merges one onto the base config per job. They therefore work unchanged
across datasets.

### Analysis and plotting

`analyses/alignment_patterns/` computes model-brain and pairwise brain-brain alignment
patterns from the scoresheets; `aggregate_results.py` drives it, adds noise-ceiling
normalization (`score`, `score_norm_lower`, `score_norm_upper`), bootstraps by subject
and runs the equivalence analysis.

`figure_0` and `figure_0_per_roi` rank the models by noise-ceiling-normalized RSA
pooled over ROIs, read from the aggregation's own pickles. `--model-order-from`
(`MODEL_ORDER_FROM` in the two `run_figure_0_*.sh` wrappers) points that ranking at
another dataset's `$SCRATCH` instead, which is how an NSD-synthetic panel is put on the
same x-axis as the NSD one. Models missing from the current results are dropped from the
axis with a warning rather than drawn as NaN bars; the relative order of the rest is
untouched. Without the flag the scripts behave exactly as before.

`CONSTANTS.py` holds canonical ROI/subject lists. `ROIS` and `SUBJECTS` describe BOLD
Moments and are what the paper's figures were produced with — they are imported as
default arguments across a dozen scripts and must not change. Dataset-aware code calls
`get_rois(dataset)` / `get_subjects(dataset)` / `get_surface_space(dataset)` instead,
and scripts expose a `--dataset {bold_moments,nsd}` option that feeds them.

### SLURM launchers

`tasks/*/launch.py` snapshot a run: they copy `multitasking/` into the output directory,
write one merged config per job, symlink the singularity containers, copy `run.sh`, and
submit. Jobs therefore run against the copied code, not the working tree — editing the
repo after launching does not affect in-flight jobs. Two container variants exist
(`multitasking_mmflow.sif`, `multitasking_mmaction2.sif`); the launcher picks
`mmaction2` for `mmaction2/*` models and `mmflow` otherwise, and both must exist under
`singularity/` before launching.

## Conventions and gotchas

- Ruff with Google-style pydocstyle. `_extern` (vendored Opt-CWM),
  `multitasking/plot_creation/` and `multitasking/analyses/alignment_pattern.py` are
  excluded from linting; `plot_creation` is older plotting helpers that
  `plotting_scripts/` imports, and does not follow the rest of the style.
- Tests requiring the real datasets skip when the data is absent (paths are hardcoded to
  this cluster's layout), so a green run on a machine without them proves less than it
  looks. Test configs live in `configs/test_configs/`.
- NSD needs three one-off preparation steps before any run —
  `multitasking.nsd.prepare_roi_betas`, `prepare_stimuli`, and the checked-in artifacts
  from `prepare_stimulus_manifest` / `prepare_trial_counts`. The README documents the
  invocations; `configs/benchmark_nsd.yaml` documents what each path means.
- NSD-synthetic needs `tasks/nsd_synthetic/download.sh`, then
  `multitasking.nsd_synthetic.prepare_stimuli` and `prepare_betas`, plus the checked-in
  counts from `prepare_trial_counts`. `multitasking/nsd_synthetic/` mirrors
  `multitasking/nsd/` but shares nothing with it except `roi_set_hash` / `cache_path`:
  the stimuli are not COCO crops, and the betas are published per trial rather than per
  image, so averaging them needs `masterordering` from the experiment design. NSD
  publishes no noise ceiling for NSD-synthetic in any space, so `prepare_betas` computes
  one from the 2-10 repetitions per image.
- `datasets_nsd_synthetic.NSDSynthetic` subclasses `datasets_nsd.NSD` — the two share
  surface space, subjects, atlas, cache format and imputation. The seams are class
  variables (`CACHE_DIRNAME`, `STIMULUS_SET`, `N_STIMULI`, the two hint strings) plus
  `_select_stimulus_ids` and `_expected_trial_counts`. `CACHE_DIRNAME` must be used both
  for the exact cache path and for the superset-cache glob: if those diverge, a synthetic
  run with a subset ROI list finds a 1000-row NSD-core cache, takes its first 220 rows
  and silently scores the model against NSD-core betas. The `stimulus_set` and row-count
  assertions in `_load_subject_betas` are what close that hole; the pre-existing ordering
  check cannot, since `1..1000` is internally valid.
- NSD-synthetic sample IDs are `syn0001`..`syn0220`, deliberately not the bare four-digit
  form. NSD-core stimuli 1-220 already own those names and their PNGs sit in a sibling
  directory, and the feature caches verify sample IDs, which are strings rather than
  pixels.
- For NSD, `fmri.stimulus_subject_ids` is deliberately separate from `fmri.sub_id`:
  keeping it pinned at all eight subjects means a benchmark run and the inter-subject
  runs it is compared against use identical stimuli. Narrowing `sub_id` alone must not
  change the stimulus set.
