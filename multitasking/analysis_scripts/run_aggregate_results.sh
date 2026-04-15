OUTPUT_SUPDIR="/your/path/to/bm/output" # should contain subdirectories, including one named output
INTERSUBJECT_DIR="/your/path/to/intersubject/output" # should contain the (1) subdirectory that holds the intersubject results
PAIRWISE_SUBJECT_DIR="/your/path/to/pairwise/subject/output" # should contain the (many) subdirectories that holds the pairwise subject results
APSIMILARITY_METRIC="pearson" # only pearson is supported for now
MAIN_OUTCOME="score" # score, score_norm_upper, score_norm_lower

python aggregate_results.py --output-supdir $OUTPUT_SUPDIR --intersubject-dir $INTERSUBJECT_DIR --pairwise-subject-dir $PAIRWISE_SUBJECT_DIR --aps-similarity-metric $APSIMILARITY_METRIC --main-outcome $MAIN_OUTCOME