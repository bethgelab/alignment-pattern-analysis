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
