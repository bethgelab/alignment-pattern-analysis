import csv
import logging
import os
import pickle
import pprint

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import pearsonr
from sklearn.linear_model import ElasticNetCV, Ridge, RidgeCV
from sklearn.metrics import make_scorer

from multitasking.utils.feature_utils import get_splithalf_xy

LOGGER = logging.getLogger(__name__)


def pearson_corr_scorer(y_true, y_pred):
    """Custom scoring function for Pearson correlation coefficient.

    Parameters:
    y_true: array-like, shape (n_samples,)
        True values.
    y_pred: array-like, shape (n_samples,)
        Predicted values.

    Returns:
    float: Pearson correlation coefficient
    """
    if y_true.ndim == 2:
        corrs = [
            pearsonr(y_true[:, i], y_pred[:, i])[0] for i in range(y_true.shape[1])
        ]
        return np.mean(corrs)  # Return a scalar
    else:
        return pearsonr(y_true, y_pred)[0]  # Scalar already


# def create_ridge_cv_input(
#     model_representations: dict,
#     brain_representations: dict,
# ):


def run_ridge_cv(
    x_train,
    x_test,
    y_train,
    y_test,
    cv_settings,
    method="ridge",
    parallelise=False,
    one_alpha_per_voxel=False,
):
    """Run RidgeCV for a single combination of model layer & ROI."""
    if cv_settings["scoring"] == "pearson":
        pearson_corr = make_scorer(pearson_corr_scorer)
        cv_settings["scoring"] = pearson_corr

    LOGGER.info("RidgeCV settings: \n")
    LOGGER.info(pprint.pformat(cv_settings))
    if one_alpha_per_voxel:
        raise NotImplementedError
        # LOGGER.info(f"Running CV for each voxel, total of {n_voxels}")
        # if parallelise:
        #     raise NotImplementedError
        # results_cv = fit_and_predict_ridge(
        #         x_train,
        #         x_test,
        #         y_train,
        #         i,
        #         cv_settings,
        #     )
        #     for i in range(n_voxels)
        # best_alphas, coefs, intercepts, best_score, preds_train, preds_test = zip(
        #     *results_cv
        # )
        # # Convert to arrays
        # best_alphas = np.array(best_alphas)  # (n_targets,)
        # coefs = np.vstack(coefs)  # (n_targets, n_features)
        # Y_pred_train = np.column_stack(preds_train)  # (n_samples, n_targets)
        # Y_pred_test = np.column_stack(preds_test)  # (n_test_samples, n_targets)
        # intercepts = np.array(intercepts)  # (n_targets,)
        # best_scores = np.array(best_score)  # (n_targets,)
        # LOGGER.info(f"Y_pred_train shape: {Y_pred_train.shape}")
        # LOGGER.info(f"Y_pred_test shape: {Y_pred_test.shape}")
        # # check how many nans
        # n_nans = np.sum(np.isnan(Y_pred_train))
        # LOGGER.info(f"Number of nans in Y_pred_train: {n_nans}")
        # n_nans = np.sum(np.isnan(Y_pred_test))
        # LOGGER.info(f"Number of nans in Y_pred_test: {n_nans}")
        # predictions = {"train": Y_pred_train, "test": Y_pred_test}
        # layer_result = {
        #     "layer": layer,
        #     "predictions": predictions,
        #     "best_scores": best_scores,
        #     "alpha_indices": None,
        #     "alphas": best_alphas,
        #     "coefs": coefs,
        #     "intercepts": intercepts,
        # }
        # mean_score = np.nanmean(best_scores)
        # std_score = np.nanstd(best_scores)

    else:
        if method == "ridge":
            LOGGER.info("Running CV, alpha_per_target = False")
            regression = RidgeCV(
                **cv_settings,
            )
        elif method == "elastic_net":
            raise NotImplementedError
            # regression = ElasticNetCV(
            #     **cv_settings,
            # )
        regression.fit(x_train, y_train)
        LOGGER.info(regression.alpha_)
        LOGGER.info(regression.coef_.shape)
        LOGGER.info(regression.best_score_.shape)
        best_coefs = regression.coef_
        best_intercepts = regression.intercept_
        mean_score = regression.best_score_.mean()
        std_score = regression.best_score_.std()
        predictions = {
            "train": x_train.dot(best_coefs.T) + best_intercepts,
            "test": x_test.dot(best_coefs.T) + best_intercepts,
        }

        layer_roi_result = {
            "predictions": predictions,
            "best_scores": regression.best_score_,
            "alpha_indices": regression.alpha_,
            "alphas": regression.alpha_,
            "coefs": regression.coef_,
            "intercepts": regression.intercept_,
        }

    LOGGER.info(f"RidgeCV done. Mean r = {mean_score:.3f} ± {std_score:.4f}")
    return layer_roi_result


def run_ridge_cv_for_layers(
    layers,
    feature_maps_redux,
    fmri_data,
    train_set_indices,
    test_set_indices,
    config,
    save_dir=None,  # Optional: path to save results,
    LOGGER=None,
):
    method = config["versa"]["voxel_encoding"]["method"]
    ridge_cv_settings = dict(config["versa"]["voxel_encoding"]["params"])
    overwrite = config["overwrite"]
    one_alpha_per_voxel = config["versa"]["voxel_encoding"]["one_alpha_per_voxel"]

    ridge_cv_settings = dict(config["versa"]["voxel_encoding"]["params"])
    if ridge_cv_settings["scoring"] == "pearson_r":
        pearson_corr = make_scorer(pearson_corr_scorer)
        ridge_cv_settings["scoring"] = pearson_corr

    LOGGER.info("RidgeCV settings: \n")
    LOGGER.info(pprint.pformat(ridge_cv_settings))
    results = dict.fromkeys(layers)
    summary_rows = []

    if save_dir is not None:
        save_dir = os.path.join(save_dir, "ridgecv_results")
        os.makedirs(save_dir, exist_ok=True)

    for layer in layers:
        if LOGGER is not None:
            LOGGER.info(f"Running Ridge CV for layer: {layer}")
        xy = get_splithalf_xy(
            feature_maps_redux[layer],
            fmri_data,
            split_indices={"train": train_set_indices, "test": test_set_indices},
        )

        x_train, y_train = xy["train"]["X"], xy["train"]["Y"]
        x_test, _ = xy["test"]["X"], xy["test"]["Y"]

        if one_alpha_per_voxel:
            if LOGGER is not None:
                LOGGER.info("Running CV for each voxel")
            LOGGER.info(f"Number of voxels: {x_train.shape[-1]}")
            results_cv = Parallel(n_jobs=-1)(
                delayed(fit_and_predict_ridge)(
                    x_train,
                    x_test,
                    xy["train"]["X"],
                    xy["test"]["X"],
                    y_train,
                    i,
                    ridge_cv_settings,
                )
                for i in range(y_train.shape[-1])
            )

            best_alphas, coefs, intercepts, best_score, preds_train, preds_test = zip(
                *results_cv, strict=False
            )

            # Convert to arrays
            best_alphas = np.array(best_alphas)  # (n_targets,)
            coefs = np.vstack(coefs)  # (n_targets, n_features)
            Y_pred_train = np.column_stack(preds_train)  # (n_samples, n_targets)
            Y_pred_test = np.column_stack(preds_test)  # (n_test_samples, n_targets)
            intercepts = np.array(intercepts)  # (n_targets,)
            best_scores = np.array(best_score)  # (n_targets,)
            LOGGER.info(f"Y_pred_train shape: {Y_pred_train.shape}")
            LOGGER.info(f"Y_pred_test shape: {Y_pred_test.shape}")
            # check how many nans
            n_nans = np.sum(np.isnan(Y_pred_train))
            LOGGER.info(f"Number of nans in Y_pred_train: {n_nans}")
            n_nans = np.sum(np.isnan(Y_pred_test))
            LOGGER.info(f"Number of nans in Y_pred_test: {n_nans}")
            predictions = {"train": Y_pred_train, "test": Y_pred_test}
            layer_result = {
                "layer": layer,
                "predictions": predictions,
                "best_scores": best_scores,
                "alpha_indices": None,
                "alphas": best_alphas,
                "coefs": coefs,
                "intercepts": intercepts,
            }
            mean_score = np.nanmean(best_scores)
            std_score = np.nanstd(best_scores)

        else:
            if method == "ridge":
                scoring = ridge_cv_settings.pop("scoring", "r2")

                regression = RidgeCV(
                    **ridge_cv_settings,
                    scoring=scoring,
                    # alphas=[10e7],#alpha_values,
                    # fit_intercept=True,
                    # # alpha_per_target=True,
                    # scoring='r2',
                    # cv=10,
                )
            elif method == "elastic_net":
                regression = ElasticNetCV(
                    **ridge_cv_settings,
                )

            regression.fit(x_train, y_train)
            LOGGER.info(regression.alpha_)
            LOGGER.info(regression.coef_.shape)
            LOGGER.info(regression.best_score_.shape)
            best_coefs = regression.coef_
            best_intercepts = regression.intercept_
            mean_score = regression.best_score_.mean()
            std_score = regression.best_score_.std()
            predictions = {
                "train": x_train.dot(best_coefs.T) + best_intercepts,
                "test": x_test.dot(best_coefs.T) + best_intercepts,
            }

            layer_result = {
                "layer": layer,
                "predictions": predictions,
                "best_scores": regression.best_score_,
                "alpha_indices": regression.alpha_,  # best_alpha_idx,
                "alphas": regression.alpha_,
                "coefs": regression.coef_,
                "intercepts": regression.intercept_,
            }

        results[layer] = layer_result
        summary_rows.append([layer, mean_score, std_score])

        LOGGER.info(
            f"Layer {layer} RidgeCV done. Mean r = {mean_score:.3f} ± {std_score:.4f}"
        )

        if save_dir is not None and (
            overwrite
            or not os.path.exists(
                os.path.join(save_dir, f"{layer}_ridgecv_results.pkl")
            )
        ):
            # Save full results
            LOGGER.info(f"Saving results for layer {layer} to {save_dir}")
            with open(
                os.path.join(save_dir, f"{layer}_ridgecv_results.pkl"), "wb"
            ) as f:
                pickle.dump(layer_result, f)

    # Save summary CSV
    if save_dir is not None:
        summary_path = os.path.join(save_dir, "ridgecv_summary.csv")

        if overwrite or not os.path.exists(summary_path):
            with open(summary_path, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["Layer", "Mean Pearson r", "Std Pearson r"])
                writer.writerows(summary_rows)
            LOGGER.info(f"Saved summary CSV to {summary_path}")

    return results


def fit_and_predict_ridge(x_train, x_test, y_train, i, cv_settings):
    y_target = y_train[:, i]

    # Step 1: Cross-validate to find best alpha
    ridge_cv = RidgeCV(**cv_settings)
    ridge_cv.fit(x_train, y_target)
    alpha = ridge_cv.alpha_

    # Step 2: Fit final model with best alpha
    ridge_final = Ridge(alpha=alpha)
    ridge_final.fit(x_train, y_target)

    # Step 3: Predict
    # print shape of xtrainorig and xtestorig

    y_pred_train = x_train.dot(ridge_final.coef_.T) + ridge_final.intercept_
    y_pred_test = x_test.dot(ridge_final.coef_.T) + ridge_final.intercept_

    best_score = ridge_cv.best_score_

    return (
        alpha,
        ridge_final.coef_,
        ridge_final.intercept_,
        best_score,
        y_pred_train,
        y_pred_test,
    )
