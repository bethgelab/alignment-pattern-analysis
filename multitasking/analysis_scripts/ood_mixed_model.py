#!/usr/bin/env python3
"""Does alignment-pattern similarity (APS) predict OOD brain alignment?

Fits the linear mixed model ``OOD ~ ID + APS + (1|ROI) + (1|family)`` with one
row per (model, ROI). ``ID`` is the model's score on NSD-core, ``OOD`` its score
on NSD-synthetic, and ``APS`` its NSD-core alignment-pattern similarity, all for
one benchmark ``--metric`` (``rsa``, ``linear_predictivity`` or
``linear_predictivity_r``; default ``rsa``). Everything
comes from the ``df_score`` pickles that ``aggregate_results`` writes, which
hold the best layer per (model, ROI), already averaged over subjects. Both
aggregations must have been run with ``--main-outcome score`` and the same
``--aps-similarity-metric``.

APS is taken from the ID aggregation on purpose: the synthetic APS is built
from the OOD scores themselves, so using it would be circular.

The 17 rows of one model are not independent, so besides the main fit the
script refits with an extra ``(1|model)`` intercept and runs a cluster
bootstrap over models.
"""
import logging
import subprocess
import warnings
from pathlib import Path
from typing import Any

import click
import numpy as np
import pandas as pd
import yaml
from scipy import stats  # type: ignore[import-untyped]

from multitasking.analyses.alignment_patterns.base import (
    SIMILARITY_METRICS,
    aggregation_pickle,
)

logger = logging.getLogger(__name__)

METRICS = ("rsa", "linear_predictivity", "linear_predictivity_r")
EXPECTED_N_ROIS = 17
EXCLUDED_ROIS = ("V6", "V6A")

# Architecture x training data. Models matching neither the exact entries nor a
# prefix rule raise an error instead of being dropped.
MODEL_FAMILIES: dict[str, str] = {
    "clip/RN50": "clip-CNN",
}
FAMILY_PREFIXES: tuple[tuple[str, str], ...] = (
    ("clip/ViT-", "clip-ViT"),
    ("taskonomy/", "taskonomy"),
    ("timm/convnext_", "timm-ConvNeXt"),
    ("timm/resnet", "timm-ResNet"),
    ("vissl/", "vissl-SimCLR"),
)

VC_FORMULA = {"roi": "0 + C(roi)", "family": "0 + C(family)"}
VC_FORMULA_WITH_MODEL = {**VC_FORMULA, "model": "0 + C(model)"}


def model_family(model: str) -> str:
    """Returns the family of a model, raising on models without a family."""
    if model in MODEL_FAMILIES:
        return MODEL_FAMILIES[model]
    for prefix, family in FAMILY_PREFIXES:
        if model.startswith(prefix):
            return family
    raise ValueError(
        f"Model {model!r} has no entry in MODEL_FAMILIES / FAMILY_PREFIXES; "
        "add one rather than dropping it."
    )


def load_best_layer(scratch: Path, aps_metric: str, metric: str) -> pd.DataFrame:
    """Loads the best-layer test rows of one metric from an aggregation.

    Args:
        scratch: ``$SCRATCH`` the aggregation wrote to.
        aps_metric: APS similarity metric the aggregation was run with.
        metric: Benchmark metric to keep, one of :data:`METRICS`.

    Returns:
        The ``df_score`` rows with ``metric == <metric>`` and ``split == "test"``,
        unique per (model, roi).
    """
    path = aggregation_pickle(scratch, "df_score", aps_metric)
    df = pd.read_pickle(path)
    df = df[(df["metric"] == metric) & (df["split"] == "test")].copy()
    if df.empty:
        raise ValueError(f"{path}: no {metric}/test rows.")
    if df.duplicated(["model", "roi"]).any():
        raise ValueError(
            f"{path}: more than one {metric}/test row per (model, roi).")
    return df.reset_index(drop=True)


def build_ood_table(
    id_df: pd.DataFrame,
    ood_df: pd.DataFrame,
    expected_n_rois: int | None = EXPECTED_N_ROIS,
) -> pd.DataFrame:
    """Joins the ID and OOD aggregations into one row per (model, ROI).

    Args:
        id_df: Output of :func:`load_best_layer` for the NSD aggregation.
        ood_df: The same for the NSD-synthetic aggregation.
        expected_n_rois: Required number of distinct ROIs, or None to skip.

    Returns:
        A frame with columns ``model, roi, family, layer, id, aps, ood``.

    Raises:
        ValueError: If layers differ, a model has no family, ROIs are wrong
            or any value is NaN.
    """
    if "stimulus_set" in ood_df.columns:
        sets = set(ood_df["stimulus_set"].dropna().unique())
        if not sets or not all(str(s).startswith("nsd_synthetic") for s in sets):
            raise ValueError(f"OOD aggregation has stimulus sets {sets}.")
    if "stimulus_set" in id_df.columns:
        sets = set(id_df["stimulus_set"].dropna().unique())
        if any(str(s).startswith("nsd_synthetic") for s in sets):
            raise ValueError(f"ID aggregation has stimulus sets {sets}.")

    merged = id_df[["model", "roi", "layer", "score_mean", "aps_score_mean"]].merge(
        ood_df[["model", "roi", "layer", "score_mean"]],
        on=["model", "roi"],
        how="inner",
        suffixes=("_id", "_ood"),
    )
    if merged.empty:
        raise ValueError("No (model, roi) pairs shared by the two aggregations.")
    mismatch = merged[merged["layer_id"] != merged["layer_ood"]]
    if len(mismatch):
        examples = mismatch[["model", "roi", "layer_id", "layer_ood"]].head(3)
        raise ValueError(
            f"Best layers differ for {mismatch['model'].nunique()} models, "
            f"e.g. {examples.values.tolist()}"
        )
    rois = set(merged["roi"])
    if rois & set(EXCLUDED_ROIS):
        raise ValueError(f"Excluded ROIs present: {rois & set(EXCLUDED_ROIS)}")
    if expected_n_rois is not None and len(rois) != expected_n_rois:
        raise ValueError(f"Expected {expected_n_rois} ROIs, found {len(rois)}.")

    table = pd.DataFrame({
        "model": merged["model"],
        "roi": merged["roi"],
        "family": [model_family(m) for m in merged["model"]],
        "layer": merged["layer_id"],
        "id": merged["score_mean_id"],
        "aps": merged["aps_score_mean"],
        "ood": merged["score_mean_ood"],
    })
    if table[["id", "aps", "ood"]].isna().any().any():
        raise ValueError("NaNs in id/aps/ood after the join.")
    n_per_model = table.groupby("model")["roi"].nunique()
    if (n_per_model != len(rois)).any():
        raise ValueError("Some models do not cover every ROI.")
    return table.sort_values(["model", "roi"]).reset_index(drop=True)


def zscore(
    table: pd.DataFrame, columns: tuple[str, ...] = ("id", "aps", "ood")
) -> pd.DataFrame:
    """Returns a copy with ``columns`` z-scored over rows."""
    out = table.copy()
    for col in columns:
        out[col] = (out[col] - out[col].mean()) / out[col].std(ddof=0)
    return out


def fit_mixed_model(
    table: pd.DataFrame,
    formula_rhs: str = "id + aps",
    reml: bool = False,
    standardize: bool = True,
    with_model_effect: bool = False,
) -> Any:
    """Fits ``ood ~ <formula_rhs>`` with crossed random intercepts.

    Crossed effects are expressed to statsmodels as one dummy group holding
    every row plus one variance component per grouping factor.

    Args:
        table: Output of :func:`build_ood_table`.
        formula_rhs: Right-hand side of the fixed-effects formula.
        reml: REML if True, else ML (needed for the likelihood-ratio test).
        standardize: Z-score id, aps and ood over rows first.
        with_model_effect: Add a ``(1|model)`` variance component.

    Returns:
        The fitted ``MixedLMResults``.
    """
    import statsmodels.formula.api as smf  # type: ignore[import-untyped]

    data = zscore(table) if standardize else table.copy()
    vc = VC_FORMULA_WITH_MODEL if with_model_effect else VC_FORMULA
    model = smf.mixedlm(
        f"ood ~ {formula_rhs}", data, groups=np.ones(len(data)),
        re_formula="0", vc_formula=vc,
    )
    return model.fit(reml=reml, method=["lbfgs"], maxiter=500)


def fit_recording_warnings(*args: Any, **kwargs: Any) -> tuple[Any, list[str]]:
    """Runs :func:`fit_mixed_model`, returning the result and its warnings."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = fit_mixed_model(*args, **kwargs)
    return result, sorted({str(w.message) for w in caught})


def coefficient_frame(result: Any, name: str, scale: str) -> pd.DataFrame:
    """Tidy table of the fixed effects of a fit with Wald z and 95% CI."""
    fe = result.fe_params.index
    ci = result.conf_int().loc[fe]
    return pd.DataFrame({
        "fit": name,
        "scale": scale,
        "term": list(fe),
        "estimate": result.fe_params.values,
        "se": result.bse_fe.values,
        "z": result.tvalues.loc[fe].values,
        "p": result.pvalues.loc[fe].values,
        "ci_low": ci[0].values,
        "ci_high": ci[1].values,
    })


def variance_components(result: Any) -> dict[str, float]:
    """Variance components of a fit, including the residual."""
    names = list(result.model.exog_vc.names)
    out = {f"var_{n}": float(v) for n, v in zip(names, result.vcomp, strict=False)}
    out["var_residual"] = float(result.scale)
    return out


def vif(
    table: pd.DataFrame, columns: tuple[str, ...] = ("id", "aps")
) -> dict[str, float]:
    """Variance inflation factor of each predictor (with an intercept)."""
    from statsmodels.stats.outliers_influence import (  # type: ignore[import-untyped]
        variance_inflation_factor,
    )

    x = np.column_stack(
        [np.ones(len(table))] + [table[c].to_numpy() for c in columns])
    return {
        c: float(variance_inflation_factor(x, i + 1))
        for i, c in enumerate(columns)
    }


def cluster_bootstrap(
    table: pd.DataFrame,
    n_bootstrap: int,
    seed: int,
    formula_rhs: str = "id + aps",
    term: str = "aps",
) -> dict[str, Any]:
    """Cluster bootstrap over models for one fixed-effect coefficient.

    Whole models (all their ROI rows) are resampled with replacement. The
    columns are z-scored once on the full table, so every resample's
    coefficient is in the same units as the main fit. Duplicated models get
    distinct labels. Fits that raise or do not converge are counted and left
    out. The main formula (no ``(1|model)``) is used.

    Returns:
        Dict with the coefficients, failure counts and the percentile 95% CI.
    """
    z = zscore(table)
    models = np.array(sorted(z["model"].unique()))
    groups = {m: g for m, g in z.groupby("model")}
    rng = np.random.default_rng(seed)
    coefs: list[float] = []
    n_failed = n_unconverged = 0
    for _ in range(n_bootstrap):
        draw = rng.choice(models, size=len(models), replace=True)
        parts = []
        for j, m in enumerate(draw):
            g = groups[m].copy()
            g["model"] = f"{m}#{j}"
            parts.append(g)
        boot = pd.concat(parts, ignore_index=True)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = fit_mixed_model(
                    boot, formula_rhs, reml=False, standardize=False)
        except Exception:  # noqa: BLE001 - statsmodels raises many types
            n_failed += 1
            continue
        if not res.converged:
            n_unconverged += 1
            continue
        coefs.append(float(res.fe_params[term]))
    arr = np.array(coefs)
    if len(arr) >= 2:
        lo, hi = (float(x) for x in np.percentile(arr, [2.5, 97.5]))
    else:
        lo = hi = float("nan")
    return {
        "coefs": arr,
        "n_requested": n_bootstrap,
        "n_used": len(arr),
        "n_exception": n_failed,
        "n_unconverged": n_unconverged,
        "ci_low": lo,
        "ci_high": hi,
        "mean": float(arr.mean()) if len(arr) else float("nan"),
    }


def likelihood_ratio_test(full: Any, null: Any) -> tuple[float, float]:
    """Returns the chi-squared statistic (1 df) and p of ``full`` vs ``null``."""
    chi2 = max(2.0 * (full.llf - null.llf), 0.0)
    return float(chi2), float(stats.chi2.sf(chi2, df=1))


def git_hash() -> str:
    """Returns the current commit hash, with ``+dirty`` for tracked changes."""
    root = Path(__file__).resolve().parents[2]
    try:
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=root, text=True).strip()
        return head + ("+dirty" if dirty else "")
    except Exception:  # noqa: BLE001
        return "unknown"


def ols_crosscheck(table: pd.DataFrame) -> pd.DataFrame:
    """OLS ``ood ~ id + aps + C(roi) + C(family)`` on z-scored data."""
    import statsmodels.formula.api as smf  # type: ignore[import-untyped]

    res = smf.ols("ood ~ id + aps + C(roi) + C(family)", zscore(table)).fit()
    ci = res.conf_int()
    terms = ["id", "aps"]
    return pd.DataFrame({
        "fit": "ols_fixed_roi_family", "scale": "standardized", "term": terms,
        "estimate": [res.params[t] for t in terms],
        "se": [res.bse[t] for t in terms],
        "z": [res.tvalues[t] for t in terms],
        "p": [res.pvalues[t] for t in terms],
        "ci_low": [ci.loc[t, 0] for t in terms],
        "ci_high": [ci.loc[t, 1] for t in terms],
    })


def run_analysis(
    table: pd.DataFrame, n_bootstrap: int, seed: int
) -> dict[str, Any]:
    """Runs every fit and returns the pieces ``main`` writes out."""
    z = zscore(table)
    ml, w_ml = fit_recording_warnings(table, "id + aps", reml=False)
    ml_null, w_null = fit_recording_warnings(table, "id", reml=False)
    reml, w_reml = fit_recording_warnings(table, "id + aps", reml=True)
    raw, w_raw = fit_recording_warnings(
        table, "id + aps", reml=False, standardize=False)
    with_model, w_model = fit_recording_warnings(
        table, "id + aps", reml=False, with_model_effect=True)
    with_model_null, _ = fit_recording_warnings(
        table, "id", reml=False, with_model_effect=True)
    chi2, p = likelihood_ratio_test(ml, ml_null)
    chi2_m, p_m = likelihood_ratio_test(with_model, with_model_null)
    boot = cluster_bootstrap(table, n_bootstrap, seed)

    boot_row = pd.DataFrame([{
        "fit": "cluster_bootstrap_ml", "scale": "standardized", "term": "aps",
        "estimate": boot["mean"],
        "ci_low": boot["ci_low"], "ci_high": boot["ci_high"],
    }])
    coefs = pd.concat([
        coefficient_frame(ml, "main_ml", "standardized"),
        coefficient_frame(reml, "main_reml", "standardized"),
        coefficient_frame(raw, "main_ml", "unstandardized"),
        coefficient_frame(with_model, "with_model_intercept_ml", "standardized"),
        ols_crosscheck(table),
        boot_row,
    ], ignore_index=True)

    fits = [
        ("id + aps + (1|roi) + (1|family)", ml),
        ("id + (1|roi) + (1|family)", ml_null),
        ("id + aps + (1|roi) + (1|family) + (1|model)", with_model),
        ("id + (1|roi) + (1|family) + (1|model)", with_model_null),
    ]
    comparison = pd.DataFrame([
        {"model": name, "reml": False, "llf": f.llf,
         "n_params": len(f.params), "converged": f.converged}
        for name, f in fits
    ])
    comparison["lrt_chi2_vs_without_aps"] = [chi2, np.nan, chi2_m, np.nan]
    comparison["lrt_p_vs_without_aps"] = [p, np.nan, p_m, np.nan]

    resid = np.asarray(ml.resid)
    diagnostics = {
        "corr_id_aps": float(np.corrcoef(z["id"], z["aps"])[0, 1]),
        "vif": vif(z),
        "residual_sd": float(resid.std()),
        "corr_resid_fitted": float(
            np.corrcoef(resid, np.asarray(ml.fittedvalues))[0, 1]),
        "variance_components_reml": variance_components(reml),
        "variance_components_ml": variance_components(ml),
        "variance_components_with_model_ml": variance_components(with_model),
    }
    warnings_by_fit = {
        "main_ml": w_ml, "null_ml": w_null, "main_reml": w_reml,
        "unstandardized_ml": w_raw, "with_model_ml": w_model,
    }
    return {
        "ml": ml, "reml": reml, "with_model": with_model, "ml_null": ml_null,
        "coefs": coefs, "comparison": comparison, "boot": boot,
        "diagnostics": diagnostics, "warnings": warnings_by_fit,
        "lrt": (chi2, p), "lrt_with_model": (chi2_m, p_m),
    }


def format_summary(table: pd.DataFrame, out: dict[str, Any]) -> str:
    """Human-readable ``summary.txt`` contents."""
    coefs = out["coefs"]
    aps = coefs[(coefs["term"] == "aps") & (coefs["scale"] == "standardized")]
    boot = out["boot"]
    families = table.drop_duplicates("model")["family"].value_counts().to_dict()
    lines = [
        f"rows={len(table)} models={table['model'].nunique()} "
        f"rois={table['roi'].nunique()}",
        f"models per family: {families}",
        f"share of negative scores: id={(table['id'] < 0).mean():.3f} "
        f"ood={(table['ood'] < 0).mean():.3f}",
        "",
        "Standardized coefficients for aps (z-scored id, aps, ood)",
        aps.to_string(index=False),
        "",
        f"LRT aps (ML, 1 df): chi2={out['lrt'][0]:.4f} p={out['lrt'][1]:.4g}",
        f"LRT aps with (1|model): chi2={out['lrt_with_model'][0]:.4f} "
        f"p={out['lrt_with_model'][1]:.4g}",
        f"Bootstrap: used {boot['n_used']}/{boot['n_requested']} "
        f"(exceptions={boot['n_exception']}, "
        f"unconverged={boot['n_unconverged']}); aps 95% percentile CI "
        f"[{boot['ci_low']:.4f}, {boot['ci_high']:.4f}]",
        "",
        "Diagnostics",
        yaml.safe_dump(out["diagnostics"], sort_keys=False),
        "Warnings per fit",
        yaml.safe_dump(out["warnings"], sort_keys=False),
        "=== Main fit (ML) ===", str(out["ml"].summary()),
        "=== REML fit ===", str(out["reml"].summary()),
        "=== With (1|model) (ML) ===", str(out["with_model"].summary()),
    ]
    return "\n".join(lines)


@click.command()
@click.option(
    "--id-scratch", required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="$SCRATCH of the NSD aggregation (source of ID and APS).")
@click.option(
    "--ood-scratch", required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="$SCRATCH of the NSD-synthetic aggregation (source of OOD).")
@click.option(
    "--output-dir", required=True, type=click.Path(path_type=Path),
    help="Persistent output directory (deliberately not $SCRATCH).")
@click.option(
    "--aps-similarity-metric", default="mse", show_default=True,
    type=click.Choice(list(SIMILARITY_METRICS)),
    help="APS metric both aggregations were run with.")
@click.option(
    "--metric", default="rsa", show_default=True,
    type=click.Choice(list(METRICS)),
    help="Benchmark metric used for ID, OOD and APS.")
@click.option("--n-bootstrap", default=1000, show_default=True, type=int)
@click.option("--seed", default=0, show_default=True, type=int)
def main(id_scratch: Path, ood_scratch: Path, output_dir: Path,
         aps_similarity_metric: str, metric: str, n_bootstrap: int,
         seed: int):
    """Fit OOD ~ ID + APS + (1|ROI) + (1|family) and write the results."""
    logging.basicConfig(level=logging.INFO)
    id_df = load_best_layer(id_scratch, aps_similarity_metric, metric)
    ood_df = load_best_layer(ood_scratch, aps_similarity_metric, metric)
    table = build_ood_table(id_df, ood_df)
    logger.info("Table: %d rows, %d models", len(table), table["model"].nunique())

    out = run_analysis(table, n_bootstrap, seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_dir / "ood_table.csv", index=False)
    out["coefs"].to_csv(output_dir / "coefficients.csv", index=False)
    out["comparison"].to_csv(output_dir / "model_comparison.csv", index=False)
    summary = format_summary(table, out)
    (output_dir / "summary.txt").write_text(summary)
    config = {
        "id_scratch": str(id_scratch),
        "ood_scratch": str(ood_scratch),
        "aps_similarity_metric": aps_similarity_metric,
        "metric": metric,
        "seed": seed,
        "n_bootstrap": n_bootstrap,
        "git_hash": git_hash(),
    }
    (output_dir / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False))
    click.echo(summary.split("Warnings per fit")[0])


if __name__ == "__main__":
    main()
