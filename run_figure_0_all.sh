#!/bin/bash
# Run figure_0 for all metric/main-outcome combinations, with and without taskonomy models.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Which dataset's canonical ROI/subject lists to use: bold_moments (default) or nsd.
DATASET="${DATASET:-bold_moments}"

# ROIs to drop entirely, comma-separated (e.g. "V6,V6A"). An alignment pattern
# is a vector over the retained ROIs, so this must match across the
# aggregation and every figure built from it.
EXCLUDE_ROIS="${EXCLUDE_ROIS:-}"

# Where to write the figures. Empty means the script's own default,
# results/boldmoments/plots/YYYYMMDD -- which is BOLD-Moments-named, so set
# this explicitly when plotting another dataset.
PLOT_PATH="${PLOT_PATH:-}"
PLOT_ARGS=()
if [ -n "$PLOT_PATH" ]; then PLOT_ARGS=(--plot-path "$PLOT_PATH"); fi

# Metrics: rsa, linear_predictivity
METRICS="rsa linear_predictivity"

# Main outcomes: score, score_norm_upper, score_norm_lower
MAIN_OUTCOMES="score_norm_lower"

echo "Metrics: $METRICS"
echo "Main outcomes: $MAIN_OUTCOMES"
echo ""

for metric in $METRICS; do
    for main_outcome in $MAIN_OUTCOMES; do
        echo "Running: metric=$metric main_outcome=$main_outcome taskonomy=False"
        python -m multitasking.plotting_scripts.figure_0 \
                --dataset "$DATASET" \
            --metric "$metric" \
            --main-outcome "$main_outcome" \
            --taskonomy False "${PLOT_ARGS[@]}" \
                --exclude-rois "$EXCLUDE_ROIS"

        echo "Running: metric=$metric main_outcome=$main_outcome taskonomy=True"
        python -m multitasking.plotting_scripts.figure_0 \
                --dataset "$DATASET" \
            --metric "$metric" \
            --main-outcome "$main_outcome" \
            --taskonomy True "${PLOT_ARGS[@]}" \
                --exclude-rois "$EXCLUDE_ROIS"
    done
done

echo ""
echo "Done. All figures saved."
