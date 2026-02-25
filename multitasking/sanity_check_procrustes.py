"""Todo move to better location. Called from benchmark.py to get preprocessed data."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from multitasking.utils.procrustes import OrthogonalProcrustes

LOGGER = logging.getLogger(__name__)


def sanity_check_procrustes(
    procr: OrthogonalProcrustes,
    model_features,
    model_features_test,
    fmri_data_roi,
    fmri_data_roi_test,
    config,
    model_name,
    layer,
    roi_name,
    n_frames=10,
    combine_frames=True,
    random=False,
):
    # Compute the similarity between frames of the same video, compared to
    # frames of different videos.
    # Plot the similarity scores for each video as a matrix of similarity scores.
    # So, for each video, we have
    #   model features: n_frames x n_features
    #   fmri data: n_frames x n_voxels
    #   The similarity is: np.trace(model_features @ R @ fmri_data.T)
    #   where R is the rotation matrix from procrustes.

    procr.check_is_fitted()

    frame_merge_strategy = config["procrustes"]["frame_merge_strategy"]

    score_function = config["procrustes"]["score_function"]
    output_dir = Path(config["output_dir"]) / "plots" / "sanity_check_procrustes"
    output_dir.mkdir(parents=True, exist_ok=True)

    n_videos_to_check = 10

    # video ids to check
    if random:
        random_ids = np.random.randint(
            0,
            model_features_test.shape[0] // n_frames,  # assuming test set is smaller
            n_videos_to_check,
        )
    else:
        start_idx = np.random.randint(
            0, model_features_test.shape[0] // n_frames - n_videos_to_check
        )
        random_ids = np.arange(start_idx, start_idx + n_videos_to_check)

    if frame_merge_strategy == "concatenate":

        def _select_videos(arr, random_ids=None):
            if random_ids is None:
                subsampled_arr = arr[: n_videos_to_check * n_frames, :]
            else:
                subsampled_arr = np.concatenate(
                    [arr[i * n_frames : (i + 1) * n_frames] for i in random_ids]
                )
            return subsampled_arr.reshape(n_videos_to_check, n_frames, -1)

        model_features_ = _select_videos(model_features, random_ids)
        model_features_test_ = _select_videos(model_features_test, random_ids)
        fmri_data_roi_ = _select_videos(fmri_data_roi, random_ids)
        fmri_data_roi_test_ = _select_videos(fmri_data_roi_test, random_ids)

        if combine_frames:
            # Consider only 5 frames as "one video" - I'd assume that
            # same-modality similarities
            # are then higher for frame sets of the same video

            combine_n_frames = 5

            def _combine_frames(arr):
                assert n_frames % combine_n_frames == 0, (
                    "n_frames must be divisible by combine_n_frames"
                )
                return arr.reshape(-1, combine_n_frames, arr.shape[2])

            model_features_ = _combine_frames(model_features_)
            model_features_test_ = _combine_frames(model_features_test_)
            fmri_data_roi_ = _combine_frames(fmri_data_roi_)
            fmri_data_roi_test_ = _combine_frames(fmri_data_roi_test_)
            n_videos_to_check = model_features_.shape[0]
    else:
        if frame_merge_strategy == "average_5":
            random_ids = np.arange(0, n_videos_to_check)
        model_features_ = model_features[random_ids, None, :]
        model_features_test_ = model_features_test[random_ids, None, :]
        fmri_data_roi_ = fmri_data_roi[random_ids, None, :]
        fmri_data_roi_test_ = fmri_data_roi_test[random_ids, None, :]

    fig, axs = plt.subplots(2, 1, figsize=(10, 10))

    similarity_matrix_manual = np.zeros((n_videos_to_check, n_videos_to_check))
    similarity_matrix = np.zeros((n_videos_to_check, n_videos_to_check))
    similarity_matrix_manual_test = np.zeros((n_videos_to_check, n_videos_to_check))
    similarity_matrix_test = np.zeros((n_videos_to_check, n_videos_to_check))
    similarity_matrix_fmri = np.zeros((n_videos_to_check, n_videos_to_check))
    similarity_matrix_model = np.zeros((n_videos_to_check, n_videos_to_check))
    similarity_matrix_model_untrafo = np.zeros((n_videos_to_check, n_videos_to_check))
    similarity_matrix_fmri_test = np.zeros((n_videos_to_check, n_videos_to_check))
    similarity_matrix_model_test = np.zeros((n_videos_to_check, n_videos_to_check))

    for i in range(n_videos_to_check):
        for j in range(n_videos_to_check):
            features_i, fmri_j = procr.transform(model_features_[i], fmri_data_roi_[j])
            if i == 0 and j == 0:
                LOGGER.info(
                    f"features_i.shape (after transformation): {features_i.shape}"
                )
                LOGGER.info(f"fmri_j.shape: {fmri_j.shape}")
            similarity_matrix_manual[i, j] = np.trace(features_i @ fmri_j.T)
            similarity_matrix[i, j] = procr.score(model_features_[i], fmri_data_roi_[j])

            features_j, fmri_i = procr.transform(model_features_[j], fmri_data_roi_[i])
            similarity_matrix_fmri[i, j] = np.trace(fmri_i @ fmri_j.T)
            similarity_matrix_model[i, j] = np.trace(features_i @ features_j.T)
            similarity_matrix_model_untrafo[i, j] = np.trace(
                model_features_[i] @ model_features_[j].T
            )

            # Test data
            features_i, fmri_j = procr.transform(
                model_features_test_[i], fmri_data_roi_test_[j]
            )
            similarity_matrix_manual_test[i, j] = np.trace(features_i @ fmri_j.T)
            similarity_matrix_test[i, j] = procr.score(
                model_features_test_[i], fmri_data_roi_test_[j]
            )

            features_j, fmri_i = procr.transform(
                model_features_test_[j], fmri_data_roi_test_[i]
            )
            similarity_matrix_fmri_test[i, j] = np.trace(fmri_i @ fmri_j.T)
            similarity_matrix_model_test[i, j] = np.trace(features_i @ features_j.T)

    assert np.all(similarity_matrix_fmri <= 1.1)
    assert np.all(similarity_matrix_fmri >= -1.1)
    assert np.all(similarity_matrix_model <= 1.1)
    assert np.all(similarity_matrix_model >= -1.1)
    assert np.all(similarity_matrix_fmri_test <= 1.1)
    assert np.all(similarity_matrix_fmri_test >= -1.1)
    assert np.all(similarity_matrix_model_test <= 1.1)
    assert np.all(similarity_matrix_model_test >= -1.1)

    assert np.allclose(similarity_matrix_manual, similarity_matrix)
    assert np.allclose(similarity_matrix_manual_test, similarity_matrix_test)

    for split_idx, (split, simmat) in enumerate(
        zip(["train", "test"], [similarity_matrix, similarity_matrix_test], strict=True)
    ):
        im = axs[split_idx].imshow(simmat)
        plt.colorbar(im, ax=axs[split_idx])
        axs[split_idx].set_title(f"Similarity matrix for {split} split")
        axs[split_idx].set_xlabel("Video ID, fmri data")  # fmri or model
        axs[split_idx].set_ylabel("Video ID, model features")

    plt.savefig(output_dir / f"{model_name}_{layer}_{roi_name}_similarity_matrix.png")
    LOGGER.info(
        f"Saved similarity matrix for score function {score_function} for "
        f"{model_name}_{layer}_{roi_name}"
        f" to {output_dir / f'{model_name}_{layer}_{roi_name}_similarity_matrix.png'}"
    )

    # also plot fmri-fmri similarity etc, as sanity check of the sanity check
    fig, axs = plt.subplots(3, 2, figsize=(15, 10))
    im = axs[0, 0].imshow(similarity_matrix_fmri)
    plt.colorbar(im, ax=axs[0, 0])
    axs[0, 0].set_title("FMRI-FMRI similarity")
    im = axs[1, 0].imshow(similarity_matrix_model)
    plt.colorbar(im, ax=axs[1, 0])
    axs[1, 0].set_title("Model-Model similarity")
    im = axs[2, 0].imshow(similarity_matrix_model_untrafo)
    plt.colorbar(im, ax=axs[2, 0])
    axs[2, 0].set_title("Model-Model similarity without applying R or normalization")
    im = axs[0, 1].imshow(similarity_matrix_fmri_test)
    plt.colorbar(im, ax=axs[0, 1])
    axs[0, 1].set_title("FMRI-FMRI similarity (test)")
    im = axs[1, 1].imshow(similarity_matrix_model_test)
    plt.colorbar(im, ax=axs[1, 1])
    axs[1, 1].set_title("Model-Model similarity (test)")
    plt.suptitle(f"{model_name} {layer} {roi_name}; video ids: {random_ids}")
    plt.savefig(
        output_dir
        / f"{model_name}_{layer}_{roi_name}_similarity_matrices_same_same.png"
    )
    LOGGER.info(
        f"Saved similarity matrix for {model_name}_{layer}_{roi_name}"
        f" to {
            output_dir
            / f'{model_name}_{layer}_{roi_name}_similarity_matrices_same_same.png'
        }"
    )
