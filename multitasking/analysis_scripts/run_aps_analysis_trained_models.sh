#!/bin/bash
# Script to run alignment pattern similarity analysis for trained models

# Configuration - modify these paths as needed
DATE_STR=$(date +'%Y%m%d')
PLOT_PATH="/mnt/lustre/work/bethge/bkr578/projects/multitasking/results/boldmoments/plots/${DATE_STR}/aps"
METRIC="rsa"
SIMILARITY_METRIC="variance_explained"
# Read ROIS from CONSTANTS.py
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROIS=$(python3 -c "import sys; sys.path.insert(0, '${SCRIPT_DIR}/..'); from CONSTANTS import ROIS; print(','.join(ROIS))")
SPLIT="test"
LOG_LEVEL="INFO"

# Run the analysis
python -m multitasking.analysis_scripts.iclr_analyses_aps_trained_models \
    --split "$SPLIT" \
    --metric "$METRIC" \
    --similarity-metric "$SIMILARITY_METRIC" \
    --rois "$ROIS" \
    --plot-path "$PLOT_PATH" \
    --save-results \
    --save-plots \
    --log-level "$LOG_LEVEL"

