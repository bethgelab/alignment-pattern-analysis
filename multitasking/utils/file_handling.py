import hashlib
import json
import logging
import os
import pickle
from pathlib import Path

LOGGER = logging.getLogger(__name__)


def cache_data(file_path, data=None, load_only=False):
    """Caches data to a file.

    If the file exists, it loads and returns the data.
    If the file does not exist and `data` is provided, it saves the data.
    """
    if os.path.exists(file_path):
        LOGGER.info(f"Loading cached data from {file_path}")
        with open(file_path, "rb") as f:
            return pickle.load(f)
    elif not load_only and data is not None:
        LOGGER.info(f"Saving data to cache at {file_path}")
        with open(file_path, "wb") as f:
            pickle.dump(data, f)
    return data


# def load_target_rdms(config, region, save_str_ext):

#     save_str = f"rdm_brain_{region}{save_str_ext}.pkl"
#     output_dir = config["output_dir"] / "rdms" / save_str

#     if output_dir.exists():
#         LOGGER.info(f"Loading cached RDM from {output_dir}")
#         with open(output_dir, "rb") as f:
#             return pickle.load(f)
#     else:
#         LOGGER.info(f"RDM file does not exist at {output_dir}")
#         return None


def load_rdms(config, str, save_str_ext):
    save_str = f"rdm_{str}{save_str_ext}.pkl"
    output_dir = Path(config["output_dir"]) / "rdms" / save_str
    if output_dir.exists():
        LOGGER.info(f"Loading cached RDM from {output_dir}")
        with open(output_dir, "rb") as f:
            return pickle.load(f)
    else:
        LOGGER.info(f"RDM file does not exist at {output_dir}")
        return None


def config_hash(config: dict, keys: list[str] | None = None) -> str:
    """Returns the hash of a config dictionary."""
    if keys is None:
        relevant_config = config
    else:
        relevant_config = {k: v for k, v in config.items() if k in keys}
    config_str = json.dumps(relevant_config, sort_keys=True)
    return hashlib.sha256(config_str.encode("utf-8")).hexdigest()[:8]


def get_scoresheet(scoresheet_file):
    if scoresheet_file.exists():
        with open(scoresheet_file, "rb") as f:
            return pickle.load(f)
    else:
        LOGGER.info(f"Scoresheet file does not exist at {scoresheet_file}")
        return []
