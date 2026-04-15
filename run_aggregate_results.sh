
DIR=$(dirname "$(realpath "$0")")
OUTPUT_SUPDIR="$DIR/precomputed_results/benchmark/output" # should contain subdirectories, including one named output
INTERSUBJECT_DIR="$DIR/precomputed_results/intersubject" # should contain the (1) subdirectory that holds the intersubject results
PAIRWISE_SUBJECT_DIR="$DIR/precomputed_results/pairwise" # should contain the (many) subdirectories that holds the pairwise subject results
APSIMILARITY_METRIC="pearson" # only pearson is supported for now
MAIN_OUTCOME="score" # score, score_norm_upper, score_norm_lower

MAIN_OUTCOMES="score score_norm_lower" # score, score_norm_upper, score_norm_lower

for main_outcome in $MAIN_OUTCOMES; do
    python -m multitasking.analysis_scripts.aggregate_results --output-supdir $OUTPUT_SUPDIR --intersubject-dir $INTERSUBJECT_DIR --pairwise-subject-dir $PAIRWISE_SUBJECT_DIR --aps-similarity-metric $APSIMILARITY_METRIC --main-outcome "$main_outcome"
done