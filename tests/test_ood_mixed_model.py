import numpy as np
import pandas as pd
import pytest

pytest.importorskip("statsmodels")

from multitasking.analysis_scripts.ood_mixed_model import (  # noqa: E402
    build_ood_table,
    fit_mixed_model,
    likelihood_ratio_test,
    load_best_layer,
)

N_ROIS = 6
MODELS = (
    [f"taskonomy/t{i}" for i in range(10)]
    + [f"clip/ViT-{i}" for i in range(4)]
    + ["clip/RN50", "timm/resnet50", "timm/convnext_a", "vissl/x"]
)


def _frames(models=MODELS, ood_layer="L"):
    rows = [(m, f"roi{r}") for m in models for r in range(N_ROIS)]
    rng = np.random.default_rng(0)
    n = len(rows)
    base = pd.DataFrame(rows, columns=["model", "roi"])
    base["layer"] = "L"
    base["score_mean"] = rng.normal(size=n)
    base["aps_score_mean"] = rng.normal(size=n)
    ood = base[["model", "roi", "layer", "score_mean"]].copy()
    ood["layer"] = ood_layer
    return base, ood


def _simulate(aps_effect, seed):
    rng = np.random.default_rng(seed)
    base, ood = _frames()
    table = build_ood_table(base, ood, expected_n_rois=N_ROIS)
    roi_eff = dict(zip(sorted(table["roi"].unique()), rng.normal(0, 1, N_ROIS), strict=False))
    fam_eff = {f: rng.normal(0, 1) for f in table["family"].unique()}
    table["id"] = rng.normal(size=len(table))
    table["aps"] = rng.normal(size=len(table))
    table["ood"] = (
        0.5 * table["id"] + aps_effect * table["aps"]
        + table["roi"].map(roi_eff) + table["family"].map(fam_eff)
        + rng.normal(0, 0.5, len(table))
    )
    return table


def test_recovers_aps_effect_and_null_gives_large_p():
    table = _simulate(aps_effect=0.4, seed=1)
    res = fit_mixed_model(table, "id + aps", reml=False, standardize=False)
    lo, hi = res.conf_int().loc["aps"]
    assert res.fe_params["aps"] > 0
    assert lo < 0.4 < hi

    null = _simulate(aps_effect=0.0, seed=2)
    full = fit_mixed_model(null, "id + aps", reml=False)
    reduced = fit_mixed_model(null, "id", reml=False)
    _, p = likelihood_ratio_test(full, reduced)
    assert p > 0.05


def test_build_ood_table_raises_on_layer_mismatch():
    base, ood = _frames(ood_layer="other")
    with pytest.raises(ValueError, match="layers differ"):
        build_ood_table(base, ood, expected_n_rois=N_ROIS)


def test_build_ood_table_raises_on_unmapped_model():
    base, ood = _frames(models=MODELS + ["mystery/model"])
    with pytest.raises(ValueError, match="mystery/model"):
        build_ood_table(base, ood, expected_n_rois=N_ROIS)


def test_load_best_layer_filters_by_requested_metric(tmp_path):
    from multitasking.analyses.alignment_patterns.base import aggregation_pickle

    base, _ = _frames(models=MODELS[:2])
    frames = []
    for metric, value in [("rsa", 1.0), ("linear_predictivity_r", 2.0)]:
        for split in ("train", "test"):
            f = base.assign(metric=metric, split=split, score_mean=value)
            frames.append(f)
    pd.concat(frames).to_pickle(aggregation_pickle(tmp_path, "df_score", "mse"))

    out = load_best_layer(tmp_path, "mse", "linear_predictivity_r")
    assert len(out) == 2 * N_ROIS
    assert (out["score_mean"] == 2.0).all() and (out["split"] == "test").all()
    with pytest.raises(ValueError, match="no linear_predictivity/test"):
        load_best_layer(tmp_path, "mse", "linear_predictivity")
