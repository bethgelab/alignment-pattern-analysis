#!/bin/bash
# Script to run alignment pattern similarity analysis for random initialization models

# Configuration - modify these paths as needed
METRIC="rsa"
SIMILARITY_METRIC="pearson"
OUTPUT_SUPDIR="/mnt/lustre/work/bethge/mtangemann/projects/multitasking_prod/output/benchmark_random_networks"
RESULTS_DIR="/mnt/lustre/work/bethge/bkr578/projects/multitasking/results/boldmoments/results/20260109"
TRAINED_MODEL_APS_PATH=$RESULTS_DIR"/model_brain_aps_$METRIC.pkl"
BRAIN_BRAIN_APS_PATH=$RESULTS_DIR"/brain_brain_aps_$METRIC.pkl"
ANALYSIS_RESULTS="/mnt/lustre/work/bethge/bkr578/projects/multitasking/results/boldmoments/equivalent_models_060126.pkl"
PLOT_PATH="/mnt/lustre/work/bethge/bkr578/projects/multitasking/results/boldmoments/plots/20260109/random_init"
MODEL_COLORS="/mnt/lustre/work/bethge/bkr578/projects/multitasking/results/boldmoments/plots/model_colors_dict.pkl"

SPLIT="test"
LOG_LEVEL="INFO"

# Read ROIS from CONSTANTS.py
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROIS=$(python3 -c "import sys; sys.path.insert(0, '${SCRIPT_DIR}/..'); from CONSTANTS import ROIS; print(','.join(ROIS))" 2>/dev/null)

# If ROIS couldn't be read from CONSTANTS, use default
if [ -z "$ROIS" ]; then
    ROIS="V1,V2,V3,V4,V8,FFC,PIT,V3A,V3B,V6,V6A,V7,IPS1,MST,MT,FST,LO1,LO2,LO3"
fi

# Run the analysis
python -m multitasking.analysis_scripts.iclr_analyses_aps_random_init \
    --split "$SPLIT" \
    --output-supdir "$OUTPUT_SUPDIR" \
    --trained-model-aps-path "$TRAINED_MODEL_APS_PATH" \
    --brain-brain-aps-path "$BRAIN_BRAIN_APS_PATH" \
    --analysis-results-path "$ANALYSIS_RESULTS" \
    --plot-path "$PLOT_PATH" \
    --model-colors-dict-path "$MODEL_COLORS" \
    --metric "$METRIC" \
    --similarity-metric "$SIMILARITY_METRIC" \
    --rois "$ROIS" \
    --no-save-results \
    --save-plots \
    --log-level "$LOG_LEVEL"

