"""Test for compute_anshksoni_metric function from benchmark.py."""

import logging
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from benedict import benedict

from multitasking.benchmark import compute_anshksoni_metric

LOGGER = logging.getLogger(__name__)

# Fix the import issue by adding anshksoni_metrics to the module before importing
# from multitasking.metrics import anshksoni_neuroaimetrics_metrics
# anshksoni_neuroaimetrics_metrics.anshksoni_metrics = (
#     anshksoni_neuroaimetrics_metrics.all_metrics
# )


def create_fake_representations(
    n_samples_train=100,
    n_samples_test=30,
    n_layers=2,
    n_rois=2,
    n_features_model=64,
    n_features_brain=50,
    similarity_strength=0.7,
    seed=42,
):
    """Create fake model and brain representations where same-ROI data is more similar.

    For each layer-ROI pair, we create brain representations that are similar to
    the corresponding model layer representations. Cross-ROI/layer pairs should
    have lower similarity.

    Args:
        n_samples_train: Number of training samples
        n_samples_test: Number of test samples
        n_layers: Number of model layers
        n_rois: Number of brain ROIs
        n_features_model: Number of features per model layer
        n_features_brain: Number of features per brain ROI
        similarity_strength: How similar same-ROI data should be (0-1)
        seed: Random seed

    Returns:
        model_representations: Dict with keys like "train_layer0", "test_layer0"
        brain_representations: Dict with keys like "train_roi0", "test_roi0"
    """
    rng = np.random.default_rng(seed)

    model_representations = {}
    brain_representations = {}

    layers = [f"layer{i}" for i in range(n_layers)]
    rois = [f"roi{i}" for i in range(n_rois)]

    # Create a shared latent representation for each layer-ROI pair
    # This ensures that matching layer-ROI pairs have higher similarity
    for layer, roi in zip(layers, rois, strict=False):
        # Shared latent representation (lower dimensional)
        latent_dim = min(n_features_model, n_features_brain) // 2

        # Train data
        latent_train = rng.normal(size=(n_samples_train, latent_dim))

        # Create model representation for this layer
        model_proj_train = rng.normal(size=(latent_dim, n_features_model))
        shared_component = similarity_strength * (latent_train @ model_proj_train)
        noise_component = (1 - similarity_strength) * rng.normal(
            size=(n_samples_train, n_features_model)
        )
        model_train = shared_component + noise_component
        model_representations[f"train_{layer}"] = model_train.astype(np.float32)

        # Create brain representation for this ROI (similar to corresponding layer)
        brain_proj_train = rng.normal(size=(latent_dim, n_features_brain))
        shared_component_brain = similarity_strength * (
            latent_train @ brain_proj_train
        )
        noise_component_brain = (1 - similarity_strength) * rng.normal(
            size=(n_samples_train, n_features_brain)
        )
        brain_train = shared_component_brain + noise_component_brain
        brain_representations[f"train_{roi}"] = brain_train.astype(np.float32)

        # Test data (similar process)
        latent_test = rng.normal(size=(n_samples_test, latent_dim))
        shared_test = similarity_strength * (latent_test @ model_proj_train)
        noise_test = (1 - similarity_strength) * rng.normal(
            size=(n_samples_test, n_features_model)
        )
        model_test = shared_test + noise_test
        model_representations[f"test_{layer}"] = model_test.astype(np.float32)

        shared_test_brain = similarity_strength * (latent_test @ brain_proj_train)
        noise_test_brain = (1 - similarity_strength) * rng.normal(
            size=(n_samples_test, n_features_brain)
        )
        brain_test = shared_test_brain + noise_test_brain
        brain_representations[f"test_{roi}"] = brain_test.astype(np.float32)

    return model_representations, brain_representations, layers, rois


@pytest.mark.parametrize(
    "metric_name",
    [
        "LinearPredictivity",
        "ReverseLinearPredictivity",
        "SymmetricLinearPredictivity",
        "PLSreg",
        "PairwiseMatching",
        "SoftMatching",
        "RSA",
        "CKA",
        "VERSA",
        "Procrustes",
        "Correlation",
    ],
)
def test_compute_anshksoni_metric_same_roi_better(metric_name):
    """Test matching layer-ROI pairs score higher than mismatched pairs.

    This tests the full pipeline including:
    - Creating representations
    - Running the metric computation
    - Saving results to files
    - Verifying same-ROI scores are better than cross-ROI scores
    """
    # Create fake data
    (
        model_reps,
        brain_reps,
        layers,
        rois,
    ) = create_fake_representations(
        n_samples_train=100,
        n_samples_test=30,
        n_layers=2,
        n_rois=2,
        similarity_strength=0.8,
        seed=42,
    )

    splits = ["train", "test"]
    subject_id = "test_subject_01"

    # Create config (use actual metric_name to match real config structure)
    config = benedict({
        "feature_extraction": {
            "model": "test_model",
        },
        "anshksoni_metrics": {
            metric_name: {
                "enabled": True,
                "normalize": True,
            }
        }
    })

    # Use temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir)

        # Run the function
        compute_anshksoni_metric(
            metric_name=metric_name,
            model_representations=model_reps,
            brain_representations=brain_reps,
            layers=layers,
            rois=rois,
            splits=splits,
            config=config,
            output_path=output_path,
            subject_id=subject_id,
        )

        # Check that output files were created
        scores_file = f"scores_anshksoni_{metric_name}_{subject_id}.csv"
        scores_path = output_path / "scoresheets" / scores_file
        assert scores_path.exists(), f"Scores file not created: {scores_path}"

        # Load and check scores
        scores_df = pd.read_csv(scores_path)

        # Should have entries for all layer-ROI-split combinations
        expected_rows = len(layers) * len(rois) * len(splits)
        assert len(scores_df) == expected_rows, (
            f"Expected {expected_rows} rows, got {len(scores_df)}"
        )

        # Check that all expected columns exist
        expected_columns = [
            "model", "layer", "roi", "split", "subject", "metric",
            "n_features_1_model", "n_features_2_brain", "score", "test_scores_list"
        ]
        for col in expected_columns:
            assert col in scores_df.columns, f"Missing column: {col}"

        # Get test scores (these are the evaluation scores)
        test_scores = scores_df[scores_df["split"] == "test"].copy()

        # Check that scores are finite
        assert all(np.isfinite(test_scores["score"])), "Some scores are not finite"

        # For same-ROI pairs (layer0-roi0, layer1-roi1), scores should be better
        # than cross-ROI pairs (layer0-roi1, layer1-roi0)
        same_roi_scores = []
        cross_roi_scores = []

        for i, layer in enumerate(layers):
            for j, roi in enumerate(rois):
                score_row = test_scores[
                    (test_scores["layer"] == layer) & (test_scores["roi"] == roi)
                ]
                if len(score_row) > 0:
                    score = score_row["score"].values[0]
                    if i == j:  # Same index means matching layer-ROI
                        same_roi_scores.append(score)
                    else:  # Cross-ROI
                        cross_roi_scores.append(score)

        # Check that we have the expected number of scores
        n_rois = len(rois)
        assert len(same_roi_scores) == n_rois, (
            f"Expected {n_rois} same-ROI scores, got {len(same_roi_scores)}"
        )
        assert len(cross_roi_scores) == n_rois * (n_rois - 1), (
            f"Expected {n_rois * (n_rois - 1)} cross-ROI scores, "
            f"got {len(cross_roi_scores)}"
        )

        # Calculate means
        mean_same_roi = np.mean(same_roi_scores)
        mean_cross_roi = np.mean(cross_roi_scores)

        # Check expectation: same-ROI scores should be better
        # For distance metrics (SoftMatching, Procrustes), lower is better
        # For similarity metrics, higher is better
        # is_distance_metric = metric_name in ["SoftMatching", "Procrustes"]

        # if is_distance_metric:
        #     # For distance metrics, same-ROI should have lower (better) scores
        #     assert mean_same_roi < mean_cross_roi, (
        #         f"Expected same-ROI distance ({mean_same_roi:.4f}) < "
        #         f"cross-ROI distance ({mean_cross_roi:.4f}) for {metric_name}"
        #     )
        # else:
        # For similarity metrics, same-ROI should have higher (better) scores
        assert mean_same_roi > mean_cross_roi, (
                f"Expected same-ROI similarity ({mean_same_roi:.4f}) > "
                f"cross-ROI similarity ({mean_cross_roi:.4f}) for {metric_name}"
        )

        # Check that model directory was created
        # Note: Model files are NOT saved (commented out in benchmark.py)
        # because they were creating 5TB of data
        model_path = output_path / metric_name / "models"
        assert model_path.exists(), f"Model directory not created: {model_path}"




if __name__ == "__main__":
    # Run a quick test manually
    LOGGER.info("Running quick manual test...")
    test_compute_anshksoni_metric_same_roi_better("CKA")
    LOGGER.info("✓ CKA test passed!")

    # test_compute_anshksoni_metric_multiple_subjects()
    # LOGGER.info("✓ Multiple subjects test passed!")

    # test_compute_anshksoni_metric_disabled()
    # LOGGER.info("✓ Disabled metric test passed!")

    LOGGER.info("\nAll manual tests passed! Run pytest for full suite.")

