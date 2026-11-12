#!/usr/bin/env bash
# Downloads the NSD-synthetic files this repository needs from the public
# Natural Scenes Dataset S3 bucket.
#
# NSD's terms of use apply; see https://naturalscenesdataset.org.
#
# What is fetched, and what is deliberately not:
#
#   * nsdsynthetic_stimuli.hdf5 holds the 220 achromatic images (dataset
#     "imgBrick", shape (220, 3, 714, 1360) uint8). The remaining 64 images of
#     the 284 are chromatic and ship as eight per-subject renderings, one per
#     subject's monitor calibration. Model features are extracted once per
#     split and reused for every subject, so per-subject stimuli cannot be
#     used without breaking the sample-ordering contract. The colour files are
#     therefore skipped.
#   * Only the fsaverage betas are fetched. That is the surface space every
#     NSD run in this repository uses.
#
# Roughly 8.4 GB in total. Transfers resume, so a re-run is cheap.
#
# Usage:
#     ./tasks/nsd_synthetic/download.sh [OUTPUT_ROOT]
#
# The default OUTPUT_ROOT matches configs/benchmark_nsd_to_synthetic.yaml.

set -euo pipefail

BUCKET_URL="https://natural-scenes-dataset.s3.amazonaws.com"
OUTPUT_ROOT="${1:-/mnt/lustre/work/bethge/bkr578/data/NSD-synthetic/raw}"
SUBJECTS=(subj01 subj02 subj03 subj04 subj05 subj06 subj07 subj08)
BETAS_DIR="nsdsyntheticbetas_fithrf_GLMdenoise_RR"

fetch() {
    # fetch <key relative to the bucket root>
    local key="$1"
    local target="${OUTPUT_ROOT}/${key}"
    mkdir -p "$(dirname "${target}")"
    echo ">>> ${key}"
    curl --fail --location --retry 5 --retry-delay 10 --continue-at - \
         --output "${target}" "${BUCKET_URL}/${key}"
}

# Experiment design: masterordering (1, 744) maps each stimulus trial onto one
# of the 284 images; the CSV names each image's class and subclass.
fetch "nsddata/experiments/nsdsynthetic/nsdsynthetic_expdesign.mat"
fetch "nsddata/experiments/nsdsynthetic/nsdsyntheticimageinformation.csv"

# Stimuli.
fetch "nsddata_stimuli/stimuli/nsdsynthetic/nsdsynthetic_stimuli.hdf5"

# Betas: one float32 (163842, 744) surface per hemisphere per subject, about
# 488 MB each. These are per-trial, not per-image; prepare_betas.py averages
# each image's repetitions using masterordering.
for subject in "${SUBJECTS[@]}"; do
    for hemi in lh rh; do
        fetch "nsddata_betas/ppdata/${subject}/fsaverage/${BETAS_DIR}/${hemi}.betas_nsdsynthetic.mgh"
    done
done

echo
echo "Done. Downloaded into ${OUTPUT_ROOT}"
du -sh "${OUTPUT_ROOT}"
