#!/bin/bash
# Script to run summary plots for BOLDMoments benchmarking results
DATE_STR=$(date +'%Y%m%d')


# Configuration - modify these paths as needed

PLOT_PATH="/mnt/lustre/work/bethge/bkr578/projects/multitasking/results/boldmoments/plots/${DATE_STR}/summary_plots"
SPLIT="test"
# Format options: png, pdf, svg (can specify multiple by adding more --format flags)
FORMAT="png"
FORMAT2="svg"
    
python -m multitasking.plot_creation.summary_plots \
    --split "$SPLIT" \
    --plot-path "$PLOT_PATH" \
    --format "$FORMAT" \

