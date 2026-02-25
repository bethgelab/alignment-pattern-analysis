"""Apply procrustes to a features and a fmri data from different regions."""

import copy
import logging
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from multitasking.sanity_check_procrustes import sanity_check_procrustes
from multitasking.utils.anshksoni_procrustes import Procrustes
from multitasking.utils.procrustes import OrthogonalProcrustes

LOGGER = logging.getLogger(__name__)


def merge_frames(data, frame_merge_strategy, fmri=True, n_frames=10):
    """Merge frames of fmri data or model features.

    For frame_merge_strategy == "average" (or "average_5"),
    fmri data (fmri=True) is averaged while model features
    (fmri=False) are flattened in the (frame, feature) axes.
    """
    _, n_voxels_all = data.shape

    if frame_merge_strategy == "concatenate":
        pass
    elif frame_merge_strategy == "average":
        if fmri:
            data = data.reshape(-1, n_frames, n_voxels_all).mean(axis=1, keepdims=False)
        else:
            data = data.reshape(-1, n_frames * n_voxels_all, order="C")
    elif frame_merge_strategy == "average_5":
        assert n_frames == 10, (
            "Only 10 frames are supported for average_5 / else rewrite this"
        )
        # average over 4 frames
        n_samples = data.shape[0] / n_frames
        assert n_samples.is_integer(), (
            f"n_samples: {n_samples} = data.shape[0] / n_frames = "
            f"{data.shape[0] / n_frames} is not an integer"
        )
        n_samples = int(n_samples)
        if fmri:
            data = data.reshape(n_samples, 5, -1, order="C")
            data = data.mean(axis=1, keepdims=False)
            data = data.reshape(n_samples * 2, -1, order="C")
        else:
            data = data.reshape(n_samples * 2, -1, order="C")
    else:
        raise ValueError(
            f"Invalid frame_merge_strategy:"
            f" {frame_merge_strategy}. Allowed values: "
            f"concatenate, average, average_5"
        )
    return data


def merge_frames_old(data, frame_merge_strategy, fmri=True):
    """Expecting format below.

    n_samples, n_frames, n_voxels_all

    Merge frames of fmri data or model features.
    For frame_merge_strategy == "average" (or "average_5"),
    fmri data (fmri=True) is averaged while model features
    (fmri=False) are flattened in the (frame, feature) axes.
    """
    n_samples, n_frames, n_voxels_all = data.shape

    if frame_merge_strategy == "concatenate":
        data = data.reshape(-1, n_voxels_all, order="C")
    elif frame_merge_strategy == "average":
        if fmri:
            data = data.mean(axis=1, keepdims=False)
        else:
            data = data.reshape(n_samples, -1, order="C", copy=False)
    elif frame_merge_strategy == "average_5":
        assert n_frames == 10, (
            "Only 10 frames are supported for average_5 / else rewrite this"
        )
        # average over 4 frames
        if fmri:
            data = (
                data.reshape(n_samples, 5, -1, order="C", copy=False)
                .mean(axis=1, keepdims=False)
                .reshape(n_samples * 2, -1, order="C", copy=False)
            )
        else:
            data = data.reshape(n_samples * 2, -1, order="C", copy=False)
    else:
        raise ValueError(
            f"Invalid frame_merge_strategy:"
            f" {frame_merge_strategy}. Allowed values: "
            f"concatenate, average, average_5"
        )
    return data


def model_str(config):
    model_str = config["feature_extraction"]["model"]
    model_str = model_str.replace("/", "_")
    return model_str


def get_scoresheet(scoresheet_file):
    """Contains a list of dicts, just like for veRSA.

    To find out whether a score has already been computed, can index on:
         - (model_str, layer, roi, split, score_function,
            projection_strategy, num_components_str)
    """
    if scoresheet_file.exists():
        with open(scoresheet_file, "rb") as f:
            return pickle.load(f)
    else:
        LOGGER.info(f"Scoresheet file does not exist at {scoresheet_file}")
        return []


def get_scoresheet_filenames(config, procrustes=True):
    output_dir = Path(config["output_dir"])
    scoresheet_path = output_dir / "scoresheets"
    scoresheet_path.mkdir(parents=True, exist_ok=True)
    suffix = ""
    if procrustes:
        suffix = "_procrustes"

    return (
        scoresheet_path / f"scoresheet{suffix}.pkl",
        scoresheet_path / f"scoresheet_procrustes_best_{model_str(config)}.pkl",
    )


def get_scoresheets(config, procrustes=True):
    # Warning: This is not parallel-safe (ie load a new unique filename always?)
    scoresheet_file, scoresheet_file_best = get_scoresheet_filenames(config, procrustes)

    scoresheet_lists = get_scoresheet(scoresheet_file)
    scoresheet_best_dict = get_scoresheet(scoresheet_file_best)
    if len(scoresheet_best_dict) == 0:
        scoresheet_best_dict = {}

    return scoresheet_lists, scoresheet_best_dict


def scoresheet_key(scoresheet):
    tup = [
        scoresheet["model_str"],
        scoresheet["model_layer"],
        scoresheet["region"],
        scoresheet["score_set"],
        scoresheet["train_type"],
        # scoresheet["n_samples"],
    ]
    if scoresheet["train_type"] == "procrustes":
        tup.extend(
            [scoresheet["procrustes_score_function"], scoresheet["projection_strategy"]]
        )
    return tuple(str(t) for t in tup)  # to sanitize None values


def scoresheet_to_dict(scoresheet_list, overwrite=False):
    return {scoresheet_key(s): s for s in scoresheet_list}
    # scoresheet_dict = {}
    # for scoresheet in scoresheet_list:
    #     key = scoresheet_key(scoresheet)
    #     if key in scoresheet_dict and not overwrite:
    #         raise ValueError(f"Score already exists at least twice for {key}")
    #     elif key in scoresheet_dict and overwrite:
    #         if ("score" in scoresheet) and (scoresheet["score"] is not None):
    #             if scoresheet["timestamp"] > scoresheet_dict[key]["timestamp"]:
    #                 # keep newest
    #                 scoresheet_dict[key] = scoresheet
    #     else:
    #         scoresheet_dict[key] = scoresheet
    # return scoresheet_dict


def compute_procrustes(
    split_model_features,
    split_brain_representation,
    layers,
    rois,
    # rois_to_indices,
    splits,
    config,
    overwrite_score=False,
    sanity_check=True,
    n_frames=10,
    test_merge_frames=False,
    use_cv_ankhsoni_prokrustes=False,
):
    """For each pair of model features and fmri data, compute the procrustes alignment."""  # noqa: E501
    if not config.get("procrustes.enabled", False):
        LOGGER.info("Orthogonal procrustes is disabled. Skipping its computation.")
        return

    is_intersubject = ( # only used for objaverse, can drop
        config["feature_extraction.model"].startswith("sub-")
        or config["feature_extraction.model"] == "all-other-subjects"
    )
    dataset = config["dataset"].get("name", "fmri_objaverse")

    if dataset == "fmri_objaverse":
        try:
            frame_merge_strategy = config["procrustes"]["frame_merge_strategy"]
        except KeyError:  # if we use it for both alignment metrics
            frame_merge_strategy = config["frame_merge_strategy"]
    feature_reduction_method = config["procrustes"]["feature_reduction_method"]
    feature_reduction_seed = config["procrustes"]["feature_reduction_seed"]
    n_components = config["procrustes"]["n_components"]
    if (n_components is not None) and (n_components <= 1):
        n_components = None
        # procrustes-feature-reduction doesnt use float values;
        # set an integer value
    if n_components is None:
        LOGGER.info(
            "No n_components specified in config. For each layer-roi pair, "
            "we will downproject to the smaller of the two dimensions."
        )

    score_function = config["procrustes"]["score_function"]
    if score_function not in ["frobenius", "angular"]:
        raise ValueError(
            f"Invalid alignment--procrustes--score_function"
            f" set in config: {score_function}"
        )

    # Initialize or load existing scoresheet
    scoresheet_lists, scoresheet_best_dict = get_scoresheets(config)
    if overwrite_score:
        scoresheet_dict = scoresheet_to_dict(scoresheet_lists, overwrite=True)

    # -------------------------------------------------
    # Preprocessing copied from veRSA pipeline (except feature reduction,
    #  since the target dimension is dependent on the fMRI ROI size,
    #  so we need to recompute it for every comparison anyways. )
    # -------------------------------------------------

    # ------------Step 1: preprocess fmri data ---------------
    # n_samples, n_frames, n_voxels_all = fmri_data.shape

    # ------------Step 2: reshape & sort model features ---------------
    for layer_index, layer in enumerate(layers):
        LOGGER.info("layer: %s", layer)
        LOGGER.info(
            "Model features shape: %s",
            split_model_features[f"train_{layer}"].shape,
        )

        # model_features_by_split = {}
        # # Split into train and test
        # for split in ["train", "test"]:
        #     model_features_by_split[split] = model_features[layer][
        #         indices_by_split[split]
        #     ]

        # ------------Step 3: compute procrustes ---------------
        # For each layer and each ROI, we get one score

        for roi_name in rois:
            # for roi_name, voxel_indices in rois_to_indices.items():
            # Initialize scoresheet and check for existing score
            train_type = "procrustes"
            if use_cv_ankhsoni_prokrustes:
                train_type += "_cv_anshksoni"
            scoresheet = {
                # lots of duplicate keys; keeping for backwards compatibility
                "score": None,
                "model_str": model_str(config),  # duplicate
                "model": config["feature_extraction"]["model"],
                "model_layer": layer,  # duplicate
                "layer": layer,
                "model_layer_index": layer_index,
                "roi": roi_name,
                "region": roi_name,  # duplicate
                "score_set": None,  # duplicate
                "split": None,
                "train_type": train_type,  # duplicate
                "metric": train_type,
                "procrustes_score_function": score_function,
                "projection_strategy": feature_reduction_method,
                "procrustes_n_components": n_components,
                "timestamp": time.time(),
                # "n_samples": n_samples_alignment_str,
                "is_intersubject": is_intersubject,
                "subject": config["fmri"]["sub_id"],
            }
            if not overwrite_score:
                key = scoresheet_key(scoresheet)
                if key in scoresheet_dict:
                    LOGGER.info(
                        f"Skipping alignment computation for {key} "
                        f"because it already exists"
                    )
                    continue

            LOGGER.info(f"Computing procrustes for {roi_name} and layer {layer}")

            # Get both feature maps for train split
            cur_model_features = split_model_features[f"train_{layer}"]
            fmri_data_roi = split_brain_representation[f"train_{roi_name}"]
            n_components_cur = min(
                n_components, cur_model_features.shape[1], fmri_data_roi.shape[1]
            )
            scoresheet["procrustes_n_components"] = n_components_cur

            LOGGER.info(
                f"Computing procrustes between model features with "
                f"feature dim: {cur_model_features.shape} and "
                f"fmri data dim: {fmri_data_roi.shape}"
            )

            # Merge or reshape frame dim
            # Remove this and the assertion when it works
            if (
                (not is_intersubject)
                and test_merge_frames
                and dataset == "fmri_objaverse"
            ):
                cur_model_features_old_method = merge_frames_old(
                    cur_model_features.reshape(
                        -1, n_frames, cur_model_features.shape[1]
                    ),
                    frame_merge_strategy,
                    fmri=False,
                )

            if dataset == "fmri_objaverse":
                cur_model_features = merge_frames(
                    cur_model_features,
                    frame_merge_strategy,
                    fmri=(not is_intersubject),
                    n_frames=n_frames,
                )
                if (not is_intersubject) and test_merge_frames:
                    assert np.allclose(
                        cur_model_features, cur_model_features_old_method
                    ), (
                        "cur_model_features and cur_model_features_old_method are not "
                        "close:"
                        f"cur_model_features shape: {cur_model_features.shape},"
                        f"cur_model_features_old_method shape: "
                        f"{cur_model_features_old_method.shape}"
                    )
                fmri_data_roi = merge_frames(
                    fmri_data_roi, frame_merge_strategy, fmri=True, n_frames=n_frames
                )
                LOGGER.info(
                    f"After merging, model features dim: {cur_model_features.shape} "
                    f"and fmri data dim: {fmri_data_roi.shape}"
                )
            elif dataset == "bold_moments":
                pass
            else:
                raise ValueError(f"Unknown dataset: {dataset}")

            if use_cv_ankhsoni_prokrustes:
                train_score, test_score = Procrustes(
                    cur_model_features, fmri_data_roi, return_similarity=True
                )
            else:
                # Compute Procrustes
                procr = OrthogonalProcrustes(
                    score_function=score_function,
                    projection_strategy=feature_reduction_method,
                    projection_seed=feature_reduction_seed,
                    n_components=n_components_cur,
                    standardize_features=config["procrustes"].get(
                        "normalize_features",
                        False,
                    ),
                )
                train_score = procr.fit(cur_model_features, fmri_data_roi)
                train_score_2 = procr.score(cur_model_features, fmri_data_roi)
                assert np.isclose(train_score, train_score_2), (
                    f"train_score: {train_score},"
                    f"train_score_2: {train_score_2} - should be close"
                )

                # Compute test score
                model_features_test = split_model_features[f"test_{layer}"]
                fmri_data_roi_test = split_brain_representation[f"test_{roi_name}"]

                if dataset == "fmri_objaverse":
                    model_features_test = merge_frames(
                        model_features_test,
                        frame_merge_strategy,
                        fmri=(not is_intersubject),
                        n_frames=n_frames,
                    )
                    fmri_data_roi_test = merge_frames(
                        fmri_data_roi_test,
                        frame_merge_strategy,
                        fmri=True,
                        n_frames=n_frames,
                    )

                test_score = procr.score(model_features_test, fmri_data_roi_test)

            LOGGER.info(f"Test score: {test_score}, train score: {train_score}")

            # Generate scoresheet entry
            for score, score_set in zip(
                [train_score, test_score], ["train", "test"], strict=True
            ):
                scoresheet_ = copy.deepcopy(scoresheet)
                scoresheet_["score"] = score
                scoresheet_["score_set"] = score_set
                scoresheet_["split"] = score_set
                key = scoresheet_key(scoresheet_)
                if key in scoresheet_dict:
                    if overwrite_score:
                        LOGGER.info(f"Overwriting score for {key}")
                        scoresheet_dict[key] = scoresheet_
                        scoresheet_lists = list(scoresheet_dict.values())
                else:
                    scoresheet_lists.append(scoresheet_)

                # check if we need to update best scoresheet
                if score_set == "test":
                    if key in scoresheet_best_dict:
                        if (
                            score_function == "angular"
                            and score < scoresheet_best_dict[key]["score"]
                        ):
                            scoresheet_best_dict[key] = scoresheet_
                        if (
                            score_function == "frobenius"
                            and score > scoresheet_best_dict[key]["score"]
                        ):
                            scoresheet_best_dict[key] = scoresheet_
                    else:
                        scoresheet_best_dict[key] = scoresheet_

            if dataset == "fmri_objaverse":
                if sanity_check and frame_merge_strategy in [
                    "concatenate",
                    "average_5",
                    "average",
                ]:
                    LOGGER.info("Sanity checking procrustes for this layer and roi...")
                    # n_frames_model = split_model_features[f"train_{layer}"].shape[1]
                    # assert n_frames_model == n_frames
                    sanity_check_procrustes(
                        procr,
                        cur_model_features,
                        model_features_test,
                        fmri_data_roi,
                        fmri_data_roi_test,
                        config,
                        model_str(config),
                        layer,
                        roi_name,
                        n_frames=n_frames,
                    )

    # Save scoresheet - could save in between as well if code is likely to fail
    scoresheet_file, scoresheet_file_best = get_scoresheet_filenames(
        config, procrustes=True
    )
    with open(scoresheet_file, "wb") as f:
        pickle.dump(scoresheet_lists, f)

    with open(scoresheet_file_best, "wb") as f:
        pickle.dump(scoresheet_best_dict, f)

    # I never look at this, commenting it out
    # scoresheet_best = [v for v in scoresheet_best_dict.values()]
    # plot_procrustes_scoresheet(scoresheet_best, config)

    # return scoresheet_lists, scoresheet_best_dict

    # Store also in new format

    scores_path = Path(config["output_dir"]) / "scoresheets" / "scores_procrustes.csv"
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    scores = pd.DataFrame(scoresheet_lists)
    scores.to_csv(scores_path, index=False)
    LOGGER.info(f"Computing Procrustes completed.\n{scores}")
