#!/bin/bash
# Run figure_1_v3 for all ROIs (from multitasking.CONSTANTS) and all metrics/main-outcomes.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Which dataset's canonical ROI/subject lists to use: bold_moments (default) or nsd.
DATASET="${DATASET:-bold_moments}"

# ROIs to drop entirely, comma-separated (e.g. "V6,V6A"). An alignment pattern
# is a vector over the retained ROIs, so this must match across the
# aggregation and every figure built from it.
EXCLUDE_ROIS="${EXCLUDE_ROIS:-}"

# Alignment-pattern similarity metric the aggregation was run with: pearson or
# mse. It is part of every pickle name, so it selects which aggregation to read.
APSIMILARITY_METRIC="${APSIMILARITY_METRIC:-pearson}"

# Where to write the figures. Empty means the script's own default,
# results/boldmoments/plots/YYYYMMDD -- which is BOLD-Moments-named, so set
# this explicitly when plotting another dataset.
PLOT_PATH="${PLOT_PATH:-}"
PLOT_ARGS=()
if [ -n "$PLOT_PATH" ]; then PLOT_ARGS=(--plot_path "$PLOT_PATH"); fi

# Pull ROIs from multitasking.CONSTANTS
ROIS=$(python3 -c "
import sys
sys.path.insert(0, '.')
from multitasking.CONSTANTS import get_rois, parse_roi_names
print(' '.join(get_rois('$DATASET', exclude=parse_roi_names('$EXCLUDE_ROIS'))))
")

# Metrics: rsa, linear_predictivity
METRICS="rsa linear_predictivity"

# Main outcomes: score, score_norm_upper, score_norm_lower
MAIN_OUTCOMES="score_norm_lower"

echo "ROIs: $ROIS"
echo "Metrics: $METRICS"
echo "Main outcomes: $MAIN_OUTCOMES"
echo ""

for metric in $METRICS; do
    for main_outcome in $MAIN_OUTCOMES; do
        for roi in $ROIS; do
            echo "Running: metric=$metric main_outcome=$main_outcome roi=$roi"
            python -m multitasking.plotting_scripts.figure_1 \
                --dataset "$DATASET" \
                --metric "$metric" \
                --main-outcome "$main_outcome" \
                --roi "$roi" "${PLOT_ARGS[@]}" \
                --exclude-rois "$EXCLUDE_ROIS" \
                --aps-similarity-metric "$APSIMILARITY_METRIC"
        done
    done
done

echo ""
echo "Done. All figures saved."
