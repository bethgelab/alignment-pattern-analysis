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


### Natural Scenes Dataset (NSD)

The benchmark also supports [NSD](https://naturalscenesdataset.org/), as a second,
independent dataset. NSD uses still images, the fsaverage surface and eight subjects, but
the same HCP-MMP parcel names as BOLD Moments, so the cross-region alignment patterns
remain comparable.

Three one-off preparation steps are needed; each writes a small artifact that every later
run reuses.

**1. fMRI betas.** Download the `fsaverage` `betas_fithrf_GLMdenoise_RR` data for subjects
1-8 (this requires accepting the NSD terms), then produce the per-subject prepared betas
and noise ceilings. Then extract the ROI vertices into a compact cache:

```bash
python -m multitasking.nsd.prepare_roi_betas \
    --fmri-path /path/to/NSD --output-dir /path/to/cache
```

This reads each subject's ~3.9 GB prepared-betas pickle once and writes ~20 MB per
subject. Build it with the full ROI list; runs that use a subset of those ROIs read the
same cache.

**2. Stimuli.** NSD's stimuli are COCO images cropped to a square and resized to 425x425,
and the crop boxes are published, so they can be reconstructed from a local COCO 2017
copy rather than downloading the 37 GB `nsd_stimuli.hdf5`:

```bash
python -m multitasking.nsd.prepare_stimuli \
    --coco-root /path/to/coco --output-dir /path/to/nsd_stimuli
```

The reconstruction is checked before anything is written: the stimulus-id mapping is
verified against the `shared1000` flag, every crop box is checked to describe a square
region, and every COCO source image is checked to resolve. Note that NSD resized with
MATLAB's `imresize`, so the result is not bit-identical to the published stimuli; this is
recorded in the `provenance.json` written alongside the images.

**3. Run it.** Point `configs/benchmark_nsd.yaml` at those three paths and run as usual:

```bash
python -m multitasking.benchmark --config configs/benchmark_nsd.yaml
```

A note on the stimulus set. NSD subjects completed different numbers of sessions, so they
saw different subsets of the 1000 shared stimuli. The usable set is the intersection over
`fmri.stimulus_subject_ids`, which is deliberately separate from `fmri.sub_id`: keeping it
fixed at all eight subjects means a benchmark run and the pairwise inter-subject runs it
is compared against use identical stimuli. With the default `min_n_trials: 2` that gives
766 stimuli, split 618 train / 148 test. Restricting both lists to the four subjects who
completed 40 sessions (`subj01`, `subj02`, `subj05`, `subj07`) instead gives all 1000,
split 800 / 200.

## Reproducing alignment pattern analysis

We provide precomputed results from running the benchmarking pipeline. Alignment pattern analysis can be performed based on these results by executing 
```bash
./run_aggregate_results.sh
```
Run figure shell scripts (```run_figure_{x}.sh```) to recreate the paper figures.

### NSD-synthetic

[NSD-synthetic](https://www.nature.com/articles/s41467-026-69345-9) adds one scan session
to the same eight NSD subjects, measuring responses to 284 deliberately non-naturalistic
images: noise, Mooney and line-drawing scenes, contrast and phase-coherence ladders,
single words, spiral gratings and chromatic noise. Same scanner, same GLMsingle pipeline,
same fsaverage surface, same HCP-MMP parcels - so it slots in as a test set and asks
whether a model's alignment pattern survives when the stimuli leave the natural-image
manifold.

It is used as exactly that: the encoding model and the model RDM are fitted on the
NSD-core train split and scored against NSD-synthetic. `configs/benchmark_nsd_to_synthetic.yaml`
is `configs/benchmark_nsd.yaml` plus a `test_split:` block that `build_dataset`
deep-merges onto `dataset` and `fmri` when it builds the test split.

**1. Download** (about 8.4 GB; NSD's terms of use apply):

```bash
./tasks/nsd_synthetic/download.sh /path/to/NSD-synthetic/raw
```

**2. Stimuli.** These are generated images rather than COCO crops, so they come straight
out of the published HDF5:

```bash
python -m multitasking.nsd_synthetic.prepare_stimuli \
    --stimuli-hdf5 /path/to/NSD-synthetic/raw/nsddata_stimuli/stimuli/nsdsynthetic/nsdsynthetic_stimuli.hdf5 \
    --image-info /path/to/NSD-synthetic/raw/nsddata/experiments/nsdsynthetic/nsdsyntheticimageinformation.csv \
    --output-dir /path/to/NSD-synthetic/stimuli
```

**3. Betas and noise ceilings.** NSD-synthetic's betas are published per *trial*, so each
image's repetitions have to be averaged using `masterordering` from the experiment design.
NSD publishes no noise ceiling for NSD-synthetic in any space, so one is computed here from
the 2-10 repetitions per image, in the same pass:

```bash
python -m multitasking.nsd_synthetic.prepare_betas \
    --fmri-path /path/to/NSD-synthetic/raw \
    --expdesign /path/to/NSD-synthetic/raw/nsddata/experiments/nsdsynthetic/nsdsynthetic_expdesign.mat \
    --atlas-dir /path/to/NSD/atlas \
    --output-dir /path/to/NSD-synthetic/cache
```

Reading the 7.8 GB of surface betas takes a few minutes and writes about 5 MB per subject.
Point `--atlas-dir` at NSD-core's atlas: the HCP-MMP annotation is subject- and
stimulus-independent, and sharing it is what keeps the two datasets' voxels aligned.

**4. Run it.**

```bash
python -m multitasking.benchmark --config configs/benchmark_nsd_to_synthetic.yaml
```

Three decisions are worth knowing about, all recorded in the `provenance.json` written
beside the stimuli.

*Only the 220 achromatic images are used.* Images 221-284 are chromatic and ship as eight
per-subject renderings, one per subject's monitor calibration. Model features are
extracted once per split and reused for every subject, so per-subject stimuli would break
the sample-ordering contract. ("Achromatic" is NSD-synthetic's own term for "no chromatic
modulation" and does not mean `R == G == B`: every pixel of these 220 lies on a single ray
through RGB space along the display white point, so they carry a uniform warm cast, which
is preserved rather than normalised away.)

*The 1360x714 frame is cropped to its central 714x714.* NSD-synthetic spans 16 degrees by
8.4 of visual field where NSD-core spans 8.4 by 8.4; cropping to the centre matches
NSD-core's field of view so that features from the two datasets are comparable, at the
cost of each stimulus's left and right periphery.

*Sample IDs are `syn0001`..`syn0220`, not `0001`..`0220`.* NSD-core stimuli 1-220 already
carry the bare four-digit names and their PNGs sit in a sibling directory. Since the
feature caches verify sample IDs, and sample IDs are strings rather than pixels, a shared
namespace would let a mis-pointed stimulus path pair natural images with synthetic betas
with every check still passing.

One caveat to keep in view: synthetic stimuli 13-32 are natural NSD scenes and
manipulations of them. If any are shared-1000 stimuli in the NSD-core *train* split then
the encoding model has seen them. The released NSD-synthetic metadata carries no COCO
identifiers, so this cannot be resolved from the data; `prepare_stimuli` logs the IDs and
takes `--exclude-id` to drop them. At most 20 of 220 stimuli.


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
