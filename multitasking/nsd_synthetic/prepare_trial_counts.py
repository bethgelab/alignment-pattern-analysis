"""Regenerates the cached NSD-synthetic repetition counts.

Run once; the resulting ~400 byte file is checked in so that building the
dataset needs neither nsdsynthetic_expdesign.mat nor scipy.

Unlike NSD-core, one vector covers every subject: all eight completed the
single NSD-synthetic session in full, so the counts follow from the design
alone.

Usage:
    python -m multitasking.nsd_synthetic.prepare_trial_counts \
        --expdesign nsdsynthetic_expdesign.mat
"""

import collections
import logging
from pathlib import Path
from typing import Any

import click
import numpy as np

from multitasking.nsd_synthetic.design import (
    NSD_SYNTHETIC_N_ACHROMATIC,
    TRIAL_COUNTS_KEY,
    TRIAL_COUNTS_PATH,
    compute_trial_counts,
    load_masterordering,
)

LOGGER = logging.getLogger(__name__)


@click.command()
@click.option(
    "--expdesign",
    "expdesign_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to nsdsynthetic_expdesign.mat.",
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
    """Derives and caches the per-image repetition counts."""
    counts = compute_trial_counts(load_masterordering(expdesign_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Widened to Any: numpy's stubs otherwise match **kwargs against savez's
    # allow_pickle parameter.
    arrays: dict[str, Any] = {TRIAL_COUNTS_KEY: counts}
    np.savez_compressed(output_path, **arrays)

    achromatic = counts[:NSD_SYNTHETIC_N_ACHROMATIC]
    LOGGER.info(
        "All images: %d, repetition histogram %s",
        len(counts),
        dict(sorted(collections.Counter(counts.tolist()).items())),
    )
    LOGGER.info(
        "Achromatic images (the scored set): %d, repetition histogram %s, "
        "%d trials",
        len(achromatic),
        dict(sorted(collections.Counter(achromatic.tolist()).items())),
        int(achromatic.sum()),
    )
    LOGGER.info("Wrote %s", output_path)


if __name__ == "__main__":
    import coloredlogs

    coloredlogs.install(fmt="%(asctime)s %(name)s %(levelname)s %(message)s")
    prepare_trial_counts()
