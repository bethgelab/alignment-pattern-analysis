#!/bin/bash
# Run figure_2 for all metric/main-outcome combinations.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Which dataset's canonical ROI list to use: bold_moments (default),
# nsd, or nsd_synthetic.
DATASET="${DATASET:-bold_moments}"

# ROIs to drop entirely, comma-separated (e.g. "V6,V6A"). Must match what
# aggregate_results was run with.
# V6 and V6A are dropped by default: pass EXCLUDE_ROIS= to keep them.
EXCLUDE_ROIS="${EXCLUDE_ROIS-V6,V6A}"

# Where to write the figures. Empty means the script's own default,
# results/boldmoments/plots/YYYYMMDD -- which is BOLD-Moments-named, so set
# this explicitly when plotting another dataset.
PLOT_PATH="${PLOT_PATH:-}"
PLOT_ARGS=()
if [ -n "$PLOT_PATH" ]; then PLOT_ARGS=(--plot_path "$PLOT_PATH"); fi

# Metrics: rsa, linear_predictivity
METRICS="rsa linear_predictivity"

# Main outcomes: score, score_norm_upper, score_norm_lower
MAIN_OUTCOMES="score_norm_lower"

echo "Metrics: $METRICS"
echo "Main outcomes: $MAIN_OUTCOMES"
echo ""

for metric in $METRICS; do
    for main_outcome in $MAIN_OUTCOMES; do
        echo "Running: metric=$metric main_outcome=$main_outcome"
        python -m multitasking.plotting_scripts.figure_2 \
            --dataset "$DATASET" \
            --metric "$metric" \
            --main-outcome "$main_outcome" \
            --exclude-rois "$EXCLUDE_ROIS" "${PLOT_ARGS[@]}"
    done
done

echo ""
echo "Done. All figures saved."
