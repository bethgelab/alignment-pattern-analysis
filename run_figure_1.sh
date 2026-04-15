#!/bin/bash
# Run figure_1_v3 for all ROIs (from multitasking.CONSTANTS) and all metrics/main-outcomes.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Pull ROIs from multitasking.CONSTANTS
ROIS=$(python3 -c "
import sys
sys.path.insert(0, '.')
from multitasking.CONSTANTS import ROIS
print(' '.join(ROIS))
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
                --metric "$metric" \
                --main-outcome "$main_outcome" \
                --roi "$roi"
        done
    done
done

echo ""
echo "Done. All figures saved."
