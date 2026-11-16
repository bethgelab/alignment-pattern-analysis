#!/bin/bash
# Fit OOD ~ ID + APS + (1|ROI) + (1|family) on two existing aggregations.
#
#   ID_SCRATCH=.../aggregated/nsd_no_v6_mse_v2 \
#   OOD_SCRATCH=.../aggregated/nsd_synthetic_no_v6_mse_v2 \
#   OUTPUT_DIR=.../analyses/ood_mixed_model_mse \
#     ./run_ood_mixed_model.sh
#
# Both aggregations must be --main-outcome score and use the same APS metric.
# OUTPUT_DIR must be persistent (not $SCRATCH). Needs `uv sync --group stats`.

set -e

DIR=$(dirname "$(realpath "$0")")
cd "$DIR"

: "${ID_SCRATCH:?set ID_SCRATCH to the NSD aggregation directory}"
: "${OOD_SCRATCH:?set OOD_SCRATCH to the NSD-synthetic aggregation directory}"
: "${OUTPUT_DIR:?set OUTPUT_DIR to a persistent output directory}"
APSIMILARITY_METRIC="${APSIMILARITY_METRIC:-mse}" # pearson or mse
METRIC="${METRIC:-rsa}" # rsa, linear_predictivity or linear_predictivity_r
N_BOOTSTRAP="${N_BOOTSTRAP:-1000}"
SEED="${SEED:-0}"

python -m multitasking.analysis_scripts.ood_mixed_model \
    --id-scratch "$ID_SCRATCH" \
    --ood-scratch "$OOD_SCRATCH" \
    --output-dir "$OUTPUT_DIR" \
    --aps-similarity-metric "$APSIMILARITY_METRIC" \
    --metric "$METRIC" \
    --n-bootstrap "$N_BOOTSTRAP" \
    --seed "$SEED"
