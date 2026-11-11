"""Regenerates the cached NSD shared-1000 repetition counts.

Run once; the resulting ~8 KB file is checked in so that building the dataset
needs neither nsd_expdesign.mat nor the prepared betas.

Usage:
    python -m multitasking.nsd.prepare_trial_counts --expdesign nsd_expdesign.mat
"""

import logging
from pathlib import Path
from typing import Any

import click
import numpy as np

from multitasking.nsd.trial_counts import (
    TRIAL_COUNTS_PATH,
    compute_shared1000_trial_counts,
)

LOGGER = logging.getLogger(__name__)


@click.command()
@click.option(
    "--expdesign",
    "expdesign_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to nsd_expdesign.mat.",
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=TRIAL_COUNTS_PATH,
    show_default=True,
    help="Where to write the cached counts.",
)
def prepare_trial_counts(expdesign_path: Path, output_path: Path) -> None:
    """Derives and caches per-subject shared-1000 repetition counts."""
    counts = compute_shared1000_trial_counts(expdesign_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Widened to Any: numpy's stubs otherwise match **kwargs against savez's
    # allow_pickle parameter.
    arrays: dict[str, Any] = dict(counts)
    np.savez_compressed(output_path, **arrays)

    for subject_id, subject_counts in sorted(counts.items()):
        LOGGER.info(
            "%s: %d stimuli seen at least twice, %d at least three times",
            subject_id,
            int((subject_counts >= 2).sum()),
            int((subject_counts >= 3).sum()),
        )
    LOGGER.info("Wrote %s", output_path)


if __name__ == "__main__":
    import coloredlogs

    coloredlogs.install(fmt="%(asctime)s %(name)s %(levelname)s %(message)s")
    prepare_trial_counts()
