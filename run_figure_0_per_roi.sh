#!/bin/bash
# Run figure_0_per_roi for all metric/main-outcome combinations, with and
# without taskonomy models. Unlike run_figure_0_all.sh, which averages over
# ROIs, this writes one benchmark panel per ROI plus an overview grid.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Which dataset's canonical ROI/subject lists to use: bold_moments
# (default), nsd, or nsd_synthetic.
DATASET="${DATASET:-bold_moments}"

# ROIs to drop entirely, comma-separated (e.g. "V6,V6A"). An alignment pattern
# is a vector over the retained ROIs, so this must match across the
# aggregation and every figure built from it.
# V6 and V6A are dropped by default: pass EXCLUDE_ROIS= to keep them.
EXCLUDE_ROIS="${EXCLUDE_ROIS-V6,V6A}"

# Where to write the figures. Empty means the script's own default,
# results/boldmoments/plots/YYYYMMDD -- which is BOLD-Moments-named, so set
# this explicitly when plotting another dataset.
PLOT_PATH="${PLOT_PATH:-}"
PLOT_ARGS=()
if [ -n "$PLOT_PATH" ]; then PLOT_ARGS=(--plot-path "$PLOT_PATH"); fi

# Rank the models on another dataset's aggregation instead of this one's, so
# the two can be read off a shared x-axis: give the $SCRATCH directory that
# aggregate_results wrote for the reference dataset. Models absent from the
# current results are dropped from the axis with a warning. Empty means rank
# on this dataset's own scores.
MODEL_ORDER_FROM="${MODEL_ORDER_FROM:-}"
ORDER_ARGS=()
if [ -n "$MODEL_ORDER_FROM" ]; then
    ORDER_ARGS=(--model-order-from "$MODEL_ORDER_FROM")
fi

# Bar order: global (figure_0's pooled ranking, panels line up) or within-roi.
ORDER="${ORDER:-global}"

# Metrics: rsa, linear_predictivity
METRICS="${METRICS:-rsa linear_predictivity}"

# Main outcomes: score, score_norm_upper, score_norm_lower
MAIN_OUTCOMES="${MAIN_OUTCOMES:-score_norm_lower}"

echo "Dataset: $DATASET"
echo "Metrics: $METRICS"
echo "Main outcomes: $MAIN_OUTCOMES"
echo "Bar order: $ORDER"
echo ""

for metric in $METRICS; do
    for main_outcome in $MAIN_OUTCOMES; do
        for taskonomy in False True; do
            echo "Running: metric=$metric main_outcome=$main_outcome taskonomy=$taskonomy"
            python -m multitasking.plotting_scripts.figure_0_per_roi \
                --dataset "$DATASET" \
                --metric "$metric" \
                --main-outcome "$main_outcome" \
                --order "$ORDER" \
                --taskonomy "$taskonomy" "${PLOT_ARGS[@]}" "${ORDER_ARGS[@]}" \
                --exclude-rois "$EXCLUDE_ROIS"
        done
    done
done

echo ""
echo "Done. All figures saved."
