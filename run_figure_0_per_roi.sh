#!/bin/bash
# Run figure_0_per_roi for all metric/main-outcome combinations, with and
# without taskonomy models. Unlike run_figure_0_all.sh, which averages over
# ROIs, this writes one benchmark panel per ROI plus an overview grid.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Which dataset's canonical ROI/subject lists to use: bold_moments (default) or nsd.
DATASET="${DATASET:-bold_moments}"

# Where to write the figures. Empty means the script's own default,
# results/boldmoments/plots/YYYYMMDD -- which is BOLD-Moments-named, so set
# this explicitly when plotting another dataset.
PLOT_PATH="${PLOT_PATH:-}"
PLOT_ARGS=()
if [ -n "$PLOT_PATH" ]; then PLOT_ARGS=(--plot-path "$PLOT_PATH"); fi

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
                --taskonomy "$taskonomy" "${PLOT_ARGS[@]}"
        done
    done
done

echo ""
echo "Done. All figures saved."
