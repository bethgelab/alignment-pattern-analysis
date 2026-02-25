import logging
from pathlib import Path

import click
import matplotlib.pyplot as plt
import seaborn as sns

from multitasking.utils.scoresheet_loading import load_scoresheets_from_parent_dir

LOGGER = logging.getLogger(__name__)


@click.command()
@click.option("--noise-ceiling-results-dir",
               "noise_ceiling_results_dir",
                 type=click.Path(exists=True, file_okay=False, path_type=Path),
                 help=("The parent directory with subfolders (eg one per subject), "
                       "with subject vs all other subjects results."))
@click.option("--metric", "metric",
               help="The metric to plot. One of: 'rsa', 'procrustes', 'versa'. "
               "Important for the file name.",
               type=str,
               default="rsa")
@click.option("--old-format", "old_format",
               help="Whether to load the results in the old format (pkl files)."
               "Needed for older procrustes results.",
               is_flag=True)
def plot_noise_ceiling_results(noise_ceiling_results_dir: Path, #results_dir: Path,
                                # output_supdir: Path,
                                metric: str, old_format: bool):
    # For each result loaded, check the config to see whether the subject is the same
    # as the one in the noise_ceiling_results_dir.
    # Load the results
    # score_df = load_scoresheets_from_parent_dir(output_supdir, filename_metric=metric,
    #                                              old_format=old_format)
    intersubject_df = load_scoresheets_from_parent_dir(noise_ceiling_results_dir,  # type: ignore  # noqa: E501
                                                       filename_metric=metric,
                                                       old_format=old_format)
    # if not old_format:
    #     intersubject_scoresheet_path = (noise_ceiling_results_dir / "scoresheets" /
    #                                     f"scores_{metric}.csv")
    # else:
    #     intersubject_scoresheet_path = (noise_ceiling_results_dir / "scoresheets" /
    #                                     f"scoresheet_{metric}.pkl")
    # intersubject_df = load_single_scoresheet(intersubject_scoresheet_path,
    #                                                  old_format=old_format)
    if intersubject_df is None:
         raise FileNotFoundError(
            f"Couldn't load intersubject scoresheet for {noise_ceiling_results_dir}"
        )
    # if len(score_df) == 0:
    #      raise FileNotFoundError(f"Couldn't load scoresheets for {output_supdir}")

    # Check the config that subjects match
    # Edit: No, when matching noise ceiling to data points, we need to match the subject

    # Visualize noise ceiling: For each ROI, plot the
    #  similarity score of matching for the same "layer" (ROI) in the "model".
    # For a single subject, there are no confidence intervals possible.

    intersubject_df_same = \
        intersubject_df[intersubject_df["layer"] == intersubject_df["roi"]]

    subjects = intersubject_df_same['subject'].unique()
    if len(subjects) == 1:
        subject = subjects[0]
        subject = f"subject {subject}"
    else:
        subject = f"{len(subjects)} subjects"
    metric = intersubject_df_same["metric"].unique()
    assert len(metric) == 1, (
        f"Expected a single metric, found: {metric}")

    splits = intersubject_df_same["split"].unique()
    fig, ax = plt.subplots(figsize=(10, 6), nrows=len(splits))

    for split_idx, split in enumerate(splits):
        split_df = intersubject_df_same[intersubject_df_same["split"] == split]
        if len(subjects) == 1:
            sns.boxplot(split_df, x="roi", y="score", ax=ax[split_idx])
        else:
            sns.boxplot(split_df, x="roi", y="score", ax=ax[split_idx])

    ax[0].set_ylabel(f"{metric[0]}")
    ax[-1].set_xlabel("ROI")
    ax[0].set_title(f"Split: {splits[0]}")
    ax[1].set_title(f"Split: {splits[1]}")

    plt.suptitle(f"Noise ceiling for {subject}, metric {metric[0]}.")
    fig_path = (
        noise_ceiling_results_dir
        / f"noise_ceiling_{metric[0]}_{subject.replace(' ', '_')}.png"
    )
    plt.tight_layout()
    plt.savefig(fig_path)

    LOGGER.info(f"Saved noise ceiling plot to {fig_path}")


    # Next,


if __name__ == "__main__":
    plot_noise_ceiling_results()

