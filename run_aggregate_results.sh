
DIR=$(dirname "$(realpath "$0")")

# aggregate_results.py writes its dataframes to $SCRATCH and the plotting
# scripts read them back from there, so $SCRATCH must point at a directory
# that outlives the run. Inside a SLURM job it is node-local and is deleted
# when the job ends; on a login node it may not exist at all. Set it
# explicitly when aggregating a dataset you intend to plot:
#
#   SCRATCH=... DATASET=nsd \
#     OUTPUT_SUPDIR=.../nsd-benchmark-20261106/output \
#     INTERSUBJECT_DIR=.../nsd-isc-loom-20261107/output \
#     PAIRWISE_SUBJECT_DIR=.../nsd-isc-pairwise-20261107/output \
#     ./run_aggregate_results.sh
#
# The three directories must each hold the per-run hash directories directly:
# INTERSUBJECT_DIR and PAIRWISE_SUBJECT_DIR are listed non-recursively.

# Which dataset's canonical ROI/subject lists to use: bold_moments (default) or nsd.
DATASET="${DATASET:-bold_moments}"

# ROIs to drop entirely, comma-separated (e.g. "V6,V6A"). An alignment pattern
# is a vector over the retained ROIs, so this must match across the
# aggregation and every figure built from it.
EXCLUDE_ROIS="${EXCLUDE_ROIS:-}"
OUTPUT_SUPDIR="${OUTPUT_SUPDIR:-$DIR/precomputed_results/benchmark/}" # should contain subdirectories of individual model runs
INTERSUBJECT_DIR="${INTERSUBJECT_DIR:-$DIR/precomputed_results/intersubject}" # should contain the (1) subdirectory that holds the intersubject results
PAIRWISE_SUBJECT_DIR="${PAIRWISE_SUBJECT_DIR:-$DIR/precomputed_results/pairwise}" # should contain the (many) subdirectories that holds the pairwise subject results
APSIMILARITY_METRIC="pearson" # only pearson is supported for now
MAIN_OUTCOME="score" # score, score_norm_upper, score_norm_lower

MAIN_OUTCOMES="score score_norm_lower" # score, score_norm_upper, score_norm_lower

for main_outcome in $MAIN_OUTCOMES; do
    python -m multitasking.analysis_scripts.aggregate_results --dataset "$DATASET" --output-supdir $OUTPUT_SUPDIR --intersubject-dir $INTERSUBJECT_DIR --pairwise-subject-dir $PAIRWISE_SUBJECT_DIR --aps-similarity-metric $APSIMILARITY_METRIC --main-outcome "$main_outcome" --exclude-rois "$EXCLUDE_ROIS"
done