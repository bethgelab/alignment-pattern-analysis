#!/bin/bash
# Run figure_0 for all metric/main-outcome combinations, with and without taskonomy models.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

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
            --metric "$metric" \
            --main-outcome "$main_outcome" \
            --taskonomy False

        echo "Running: metric=$metric main_outcome=$main_outcome taskonomy=True"
        python -m multitasking.plotting_scripts.figure_0 \
            --metric "$metric" \
            --main-outcome "$main_outcome" \
            --taskonomy True
    done
done

echo ""
echo "Done. All figures saved."
