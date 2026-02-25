"""Feel free to move elsewhere if you find a better place for this."""

import logging
import os
import pickle
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger(__name__)



def load_single_scoresheet(fpath: Path, old_format: bool = False):
            if fpath.exists():
                if old_format:
                    with open(fpath, "rb") as f:
                        scoresheet = pickle.load(f)
                        assert len(scoresheet) == 1, "Expected a single scoresheet"
                        scoresheet = scoresheet[0]
                else:
                    scoresheet = pd.read_csv(fpath)
            else:
                LOGGER.warning(f"Couldn't load scoresheet for run {fpath}")
                return None
            return scoresheet


def filename_from_metric(metric: str, old_format: bool = False):
    if old_format:
        return f"scoresheet_{metric}.pkl"
    else:
        return f"scores_{metric}.csv"


def load_scoresheets_from_parent_dir(parent_dir: str | Path,
                                     filename: str | None = None,
                                     filename_metric: str = "rsa",
                                     old_format: bool = False,
                                     LOGGER: logging.Logger = None): # type: ignore[assignment]
    """Load scoresheets from a parent directory.

    You can pass either the filename itself, or the metric and it will
    generate the filename from that (and the old_format flag).

    old_format: todo deprecated as soon as you know the new format works
    for procrustes.

    Returns:
        pd.DataFrame: A DataFrame containing the scoresheets.
    """
    LOGGER = LOGGER or logging.getLogger(__name__)
    output_dirs = os.listdir(parent_dir)
    # LOGGER.info(f"Loading scoresheets from {parent_dir}")
    # LOGGER.info(f"Output dirs: {output_dirs}")
    all_scoresheets: list[pd.DataFrame] = []
    n = 0

    if len(output_dirs) == 0:
        output_dirs = os.listdir(parent_dir)
    all_scoresheets = []
    n = 0
    if filename is None:
        filename = filename_from_metric(filename_metric, old_format)

    # filename = "scoresheet.pkl" if not procrustes else "scoresheet_procrustes.pkl"
    # LOGGER.info(f"Loading scoresheets with filename: {filename}")
    for output_dir in output_dirs: # output_dirs contain the hashes (or are the hashes)
        LOGGER.info(f"Loading scoresheet for run {output_dir}")
        # os.listdir(os.path.join(parent_dir, output_dir))
        scores_dir = Path(os.path.join(parent_dir, output_dir, "scoresheets"))
        scorefiles = [f for f in Path(scores_dir).glob("*.csv")]
        for fpath in scorefiles:
            scoresheet = pd.read_csv(fpath)
            scoresheet["run"] = output_dir
            all_scoresheets.append(scoresheet)
            n += 1


    if len(all_scoresheets) >= 1 and old_format:
        # Step 1: Load the list of dicts into a DataFrame
        df = pd.DataFrame(all_scoresheets)
        LOGGER.info("Step 1: Loaded scoresheets into DataFrame")
        LOGGER.info(f"Loaded {len(df)} rows from {n} runs.")

        # Step 2: Rename or drop fields for clarity (optional)
        if any([
            col not in df.columns for col in ["roi", "metric", "split", "model", "layer"
        ]]):
          df = df.rename(
            columns={
                "region": "roi",
                "train_type": "metric",
                "score_set": "split",
                "model_str": "model",
                "layer_str": "layer",
            }
        )
        else:
            for col in ["region", "train_type", "score_set", "model_str", "layer_str"]:
                if col in df.columns:
                    df = df.drop(columns=[col])

        LOGGER.info("Step 2: Renamed/dropped columns for clarity")

    elif len(all_scoresheets) >= 1 and (not old_format):
        df = pd.concat(all_scoresheets)
    else:
        LOGGER.warning(f"No scoresheets loaded for {parent_dir}")
        return None
    return df
