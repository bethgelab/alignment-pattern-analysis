#!/bin/bash
# Script to run alignment pattern similarity analysis for trained models

# Configuration - modify these paths as needed
OUTPUT_SUPDIR="/mnt/lustre/work/bethge/mwe467/taskonomy/multitasking/output/bold_moments/bm"
INTERSUBJECT_DIR="/mnt/lustre/work/bethge/bkr857/projects/multitasking/output/bold_moments_intersubject_redo_larger_alpha_range/output/"
PAIRWISE_SUBJECT_DIR="/mnt/lustre/work/bethge/bkr857/projects/multitasking/output/bold_moments_intersubject_pairwise_rsa_lp_5x/output/"
ANALYSIS_RESULTS="/mnt/lustre/work/bethge/bkr578/projects/multitasking/results/boldmoments/equivalent_models_060126.pkl"
PLOT_PATH="/mnt/lustre/work/bethge/bkr578/projects/multitasking/results/boldmoments/plots/20260109/aps"
RESULTS_PATH="/mnt/lustre/work/bethge/bkr578/projects/multitasking/results/boldmoments/results/20260109"
MODEL_COLORS="/mnt/lustre/work/bethge/bkr578/projects/multitasking/results/boldmoments/plots/model_colors_dict.pkl"
METRIC="rsa"
SIMILARITY_METRIC="pearson"
# Read ROIS from CONSTANTS.py
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROIS=$(python3 -c "import sys; sys.path.insert(0, '${SCRIPT_DIR}/..'); from CONSTANTS import ROIS; print(','.join(ROIS))")
SPLIT="test"
LOG_LEVEL="INFO"

# Run the analysis
python -m multitasking.analysis_scripts.iclr_analyses_aps_trained_models \
    --split "$SPLIT" \
    --output-supdir "$OUTPUT_SUPDIR" \
    --pairwise-subject-dir "$PAIRWISE_SUBJECT_DIR" \
    --metric "$METRIC" \
    --similarity-metric "$SIMILARITY_METRIC" \
    --rois "$ROIS" \
    --analysis-results-path "$ANALYSIS_RESULTS" \
    --plot-path "$PLOT_PATH" \
    --results-path "$RESULTS_PATH" \
    --model-colors-dict-path None \
    --save-results \
    --save-plots \
    --log-level "$LOG_LEVEL"

