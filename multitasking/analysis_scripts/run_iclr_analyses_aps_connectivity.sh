#!/bin/bash
# Script to run alignment pattern similarity analysis with connectivity-based patterns

# Configuration - modify these paths as needed
METRIC="rsa"
OTHER_METRIC="linear_predictivity"
SIMILARITY_METRIC="pearson"
RESULTS_DIR="/mnt/lustre/work/bethge/bkr578/projects/multitasking/results/boldmoments/results/20260109"
TRAINED_MODEL_APS=$RESULTS_DIR"/model_brain_aps_$METRIC.pkl"
BRAIN_BRAIN_APS=$RESULTS_DIR"/brain_brain_aps_$METRIC.pkl"
BRAIN_BRAIN_APS_OTHER_METRIC=$RESULTS_DIR"/brain_brain_aps_$OTHER_METRIC.pkl"
CONNECTIVITY_APS="/mnt/lustre/work/bethge/bkr578/data/brainlife/processed/connectivity_aps.pkl"
ANALYSIS_RESULTS="/mnt/lustre/work/bethge/bkr578/projects/multitasking/results/boldmoments/equivalent_models_060126.pkl"
PLOT_PATH="/mnt/lustre/work/bethge/bkr578/projects/multitasking/results/boldmoments/plots/20260109/connectivity"
MODEL_COLORS="/mnt/lustre/work/bethge/bkr578/projects/multitasking/results/boldmoments/plots/model_colors_dict.pkl"
RANDOM_CONNECTIVITY="/mnt/lustre/work/bethge/bkr578/data/brainlife/processed/random"
# Read ROIS from CONSTANTS.py
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROIS=$(python3 -c "import sys; sys.path.insert(0, '${SCRIPT_DIR}/..'); from CONSTANTS import ROIS; print(','.join(ROIS))")
LOG_LEVEL="INFO"

# Run the analysis
python -m multitasking.analysis_scripts.iclr_analyses_aps_connectivity \
    --trained-model-aps-path "$TRAINED_MODEL_APS" \
    --brain-brain-aps-path "$BRAIN_BRAIN_APS" \
    --brain-brain-aps-other-metric "$BRAIN_BRAIN_APS_OTHER_METRIC" \
    --connectivity-aps-path "$CONNECTIVITY_APS" \
    --analysis-results-path "$ANALYSIS_RESULTS" \
    --plot-path "$PLOT_PATH" \
    --model-colors-dict-path "$MODEL_COLORS" \
    --metric "$METRIC" \
    --other-metric "$OTHER_METRIC" \
    --similarity-metric "$SIMILARITY_METRIC" \
    --rois "$ROIS" \
    --random-connectivity-aps-path "$RANDOM_CONNECTIVITY" \
    --save-plots \
    --log-level "$LOG_LEVEL"

