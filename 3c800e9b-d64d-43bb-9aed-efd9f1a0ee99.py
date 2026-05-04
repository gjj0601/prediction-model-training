from __future__ import annotations

from pathlib import Path
import csv
import yaml
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from sksurv.nonparametric import CensoringDistributionEstimator

from src.data_io_surv import load_survival_datasets, ensure_dir
from src.model_surv import train_coxnet, pick_alpha_by_auc, set_alpha, predict_risk
from src.evaluation_surv import survival_metrics_bundle, bootstrap_auc_ci  # <- 新增 bootstrap_auc_ci
from src.visualization_surv import save_four_figures_pan_style


# ---------- helpers ----------
def _get_event_time_from_surv(y_struct) -> tuple[np.ndarray, np.ndarray]:
    names = y_struct.dtype.names
    event = np.asarray(y_struct[names[0]], dtype=bool)
    time = np.asarray(y_struct[names[1]], dtype=float)
    return event, time


def _clip_times_to_valid_range(y_train, y_test, times: np.ndarray) -> np.ndarray:
    """
    昨天版本的裁剪逻辑：用 train 最大“事件时间”与 test 最大随访时间取交集。
    """
    e_tr, t_tr = _get_event_time_from_surv(y_train)
    e_te, t_te = _get_event_time_from_surv(y_test)

    train_max_event = float(np.max(t_tr[e_tr])) if np.any(e_tr) else float(np.max(t_tr))
    test_max_time = float(np.max(t_te))
    upper = min(train_max_event, test_max_time)

    times = np.asarray(times, dtype=float)
    return times[times <= upper]


def roc_curve_ipcw_dynamic(
    *,
    censor_est: CensoringDistributionEstimator,
    y_test,
    risk_score: np.ndarray,
    t0: float,
    n_thresholds: int = 200,
) -> dict:
    """
    Cumulative/Dynamic ROC(t0) with IPCW:
      cases: event & time <= t0
      controls: time > t0
    """
    risk_score = np.asarray(risk_score, dtype=float)
    event, time = _get_event_time_from_surv(y_test)

    cases = event & (time <= t0)
    controls = time > t0
    if cases.sum() == 0 or controls.sum() == 0:
        return {"fpr": np.array([0.0, 1.0]), "tpr": np.array([0.0, 1.0])}

    w_case = np.asarray(censor_est.predict_ipcw(y_test[cases]), dtype=float)

    G_t0 = float(np.clip(censor_est.predict_proba(np.asarray([t0], dtype=float))[0], 1e-6, 1.0))
    w_ctrl = np.full(controls.sum(), 1.0 / G_t0, dtype=float)

    r_case = risk_score[cases]
    r_ctrl = risk_score[controls]

    qs = np.linspace(0.0, 1.0, n_thresholds)
    thresholds = np.quantile(risk_score, 1.0 - qs)

    denom_tpr = float(np.sum(w_case))
    denom_fpr = float(np.sum(w_ctrl))

    tpr, fpr = [], []
    for thr in thresholds:
        tpr.append(float(np.sum(w_case * (r_case >= thr)) / denom_tpr))
        fpr.append(float(np.sum(w_ctrl * (r_ctrl >= thr)) / denom_fpr))

    return {"fpr": np.asarray(fpr), "tpr": np.asarray(tpr)}


def dump_metrics_csv(
    path: Path,
    model_label: str,
    metrics: dict,
    alpha: float | None,
    *,
    test_seed: int,
    split_seed: int | None,
) -> None:
    """Write scalar metrics for one model."""
    path.parent.mkdir(parents=True, exist_ok=True)

    cindex_val = metrics["cindex"]["cindex"]
    auc_times = metrics["auc_horizons"]["times"]
    auc_vals = metrics["auc_horizons"]["auc"]

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", model_label])
        w.writerow(["test_seed", test_seed])
        w.writerow(["split_seed_train_val", "" if split_seed is None else split_seed])
        w.writerow(["alpha", "" if alpha is None else alpha])
        w.writerow(["cindex_test", cindex_val])
        for t, a in zip(auc_times, auc_vals):
            w.writerow([f"auc_t{t:g}y", float(a)])


# ---------- main ----------
def main(cfg_path: str = "config/default_surv_dm.yaml") -> None:
    cfg = yaml.safe_load(open(cfg_path, "r", encoding="utf-8"))

    # 可选：CI 参数（不写 YAML 也行）
    ci_cfg = cfg.get("eval", {}).get("auc_ci", {})
    ci_enabled = bool(ci_cfg.get("enabled", True))
    ci_n_boot = int(ci_cfg.get("n_boot", 1000))
    ci_seed = int(ci_cfg.get("seed", 2026))

    # 1) Load once (common y + per-model X)
    X_by_model, y_all, meta, df_core = load_survival_datasets(cfg)
    event_col = cfg["data"]["event_col"]

    n = df_core.shape[0]
    idx_all = np.arange(n)

    # model order + pretty labels
    model_keys = ["base", "base_plus_protein", "base_plus_aip"]
    pretty = {
        "base": "SCORE2",
        "base_plus_protein": "SCORE2+Biomarkers",
        "base_plus_aip": "SCORE2+Biomarkers+Log-profile",
    }

    # 2) Fixed test split ONCE (same as yesterday)
    strat_all = df_core[event_col].astype(int).to_numpy()
    use_val = bool(cfg["train"].get("use_validation", False))

    test_seed = int(cfg["train"].get("test_random_state", cfg["train"].get("random_state", 42)))
    idx_pool, idx_te = train_test_split(
        idx_all,
        test_size=cfg["train"]["test_size"],
        random_state=test_seed,
        stratify=strat_all,
    )

    y_pool = y_all[idx_pool]
    y_te = y_all[idx_te]

    # 3) Prepare evaluation times using y_pool as survival_train (exactly as yesterday)
    horizons = np.asarray(cfg["eval"]["horizons_years"], dtype=float)
    grid = cfg["eval"]["auc_time_grid"]
    time_grid = np.arange(grid["start"], grid["stop"] + 1e-9, grid["step"], dtype=float)

    horizons_use = _clip_times_to_valid_range(y_pool, y_te, horizons)
    time_grid_use = _clip_times_to_valid_range(y_pool, y_te, time_grid)
    if len(horizons_use) == 0:
        raise ValueError("No valid horizons after clipping. Check follow-up time range and config horizons.")

    # 4) Output dirs
    here = Path(__file__).resolve()
    base_dir = here.parents[1] if here.parent.name == "experiments" else here.parent
    figures_dir = (base_dir / cfg["outputs"]["figures_dir"]).resolve()
    tables_dir = (base_dir / cfg["outputs"]["tables_dir"]).resolve()
    ensure_dir(figures_dir)
    ensure_dir(tables_dir)

    # 5) Fit censoring estimator once from y_pool (same as yesterday)
    censor_est = CensoringDistributionEstimator().fit(y_pool)

    split_seeds = cfg["train"].get("split_seeds", {})
    default_seed = int(cfg["train"].get("random_state", 42))

    results = {}
    roc_curves_by_t = {float(t): [] for t in horizons_use}
    auc_over_time_series = []

    # 6) Fit three models
    for key in model_keys:
        if key not in X_by_model:
            raise KeyError(f"Missing model key in X_by_model: {key}")

        seed = int(split_seeds.get(key, default_seed))

        if use_val:
            strat_pool = df_core.iloc[idx_pool][event_col].astype(int).to_numpy()
            idx_tr, idx_val = train_test_split(
                idx_pool,
                test_size=cfg["train"]["val_size"],
                random_state=seed,
                stratify=strat_pool,
            )
            y_tr = y_all[idx_tr]
            y_val = y_all[idx_val]
        else:
            idx_tr, idx_val = idx_pool, None
            y_tr, y_val = y_all[idx_tr], None

        X = X_by_model[key]
        X_tr = X.iloc[idx_tr]
        X_te = X.iloc[idx_te]

        # train bundle
        bundle = train_coxnet(X_tr, y_tr, meta[key], cfg)

        # alpha selection (EXACTLY as yesterday: y_train=y_pool + horizons_use)
        best_alpha = None
        alpha_tbl = None
        if use_val:
            X_val = X.iloc[idx_val]
            best_alpha, alpha_tbl = pick_alpha_by_auc(
                bundle=bundle,
                X_train=X_tr,
                y_train=y_pool,
                X_val=X_val,
                y_val=y_val,
                horizons_years=list(horizons_use),
            )
            bundle = set_alpha(bundle, best_alpha)
            alpha_tbl.to_csv(tables_dir / f"alpha_path_{key}.csv", index=False)

        # predict risk on fixed test
        risk_te = predict_risk(bundle, X_te)

        # metrics (EXACTLY as yesterday: y_train=y_pool + horizons_use/time_grid_use)
        metrics = survival_metrics_bundle(
            y_train=y_pool,
            y_test=y_te,
            risk_test=risk_te,
            horizons=horizons_use,
            auc_time_grid=time_grid_use,
            cindex_method="harrell",
        )

        # ---- NEW: bootstrap CI for AUC@horizons_use (does NOT change point estimate) ----
        auc_ci = None
        if ci_enabled:
            auc_ci = bootstrap_auc_ci(
                y_train=y_pool,
                y_test=y_te,
                risk_score_test=risk_te,
                times=horizons_use,
                n_boot=ci_n_boot,
                seed=ci_seed,
            )
            ci_df = pd.DataFrame({
                "time": auc_ci["times"],
                "auc": auc_ci["auc"],
                "ci_low": auc_ci["ci_low"],
                "ci_high": auc_ci["ci_high"],
                "valid_boot": auc_ci["valid_boot_per_time"],
            })
            ci_df.to_csv(tables_dir / f"auc_ci_{key}.csv", index=False)

        # store bundle for SHAP + store auc_ci for summary
        results[key] = {
            "bundle": bundle,
            "alpha": best_alpha,
            "metrics": metrics,
            "risk_te": risk_te,
            "split_seed": seed,
            "auc_ci": auc_ci,
        }

        dump_metrics_csv(
            tables_dir / f"metrics_{key}.csv",
            model_label=pretty[key],
            metrics=metrics,
            alpha=best_alpha,
            test_seed=test_seed,
            split_seed=seed if use_val else None,
        )

        auc_over_time_series.append(
            {
                "label": pretty[key],
                "times": metrics["auc_curve"]["times"],
                "auc": metrics["auc_curve"]["auc"],
                "cindex": float(metrics["cindex"]["cindex"]),
            }
        )

    # ===== interpretability (SCORE2-PRO / Base+Protein) =====
    if cfg.get("interpretability", {}).get("enabled", False):
        from src.interpretability_shap import shap_rank_linear_coxnet, plot_top_shap_bar

        model_key = cfg["interpretability"].get("model_key", "base_plus_protein")
        top_k = int(cfg["interpretability"].get("top_k", 15))

        if model_key not in results:
            raise KeyError(f"interpretability.model_key '{model_key}' not in results: {list(results.keys())}")

        bundle_int = results[model_key]["bundle"]
        X_pool_int = X_by_model[model_key].iloc[idx_pool]
        X_test_int = X_by_model[model_key].iloc[idx_te]

        tbl = shap_rank_linear_coxnet(bundle_int, X_pool_int, X_test_int, top_k=top_k)
        tbl.to_csv(tables_dir / f"shap_top{top_k}_{model_key}.csv", index=False)
        plot_top_shap_bar(tbl, top_k, figures_dir / f"shap_top{top_k}_{model_key}.png")

    # 7) Build ROC(t) curves at horizons_use (same as yesterday)
    for t0 in horizons_use:
        for key in model_keys:
            m = results[key]["metrics"]
            times = np.asarray(m["auc_horizons"]["times"], dtype=float)
            aucs = np.asarray(m["auc_horizons"]["auc"], dtype=float)
            idx = int(np.where(np.isclose(times, t0))[0][0])
            auc_t0 = float(aucs[idx])

            roc = roc_curve_ipcw_dynamic(
                censor_est=censor_est,
                y_test=y_te,
                risk_score=results[key]["risk_te"],
                t0=float(t0),
                n_thresholds=200,
            )
            roc_curves_by_t[float(t0)].append(
                {"label": pretty[key], "fpr": roc["fpr"], "tpr": roc["tpr"], "auc": auc_t0}
            )

    # 8) Save figures (NO forced 15y; exactly like yesterday)
    save_four_figures_pan_style(
        out_dir=figures_dir,
        roc_curves_by_t=roc_curves_by_t,
        auc_over_time_series=auc_over_time_series,
        horizons=list(horizons_use),
        prefix="",
        show=False,
        dpi=150,
    )

    # 9) Summary table across models (+ optional CI columns)
    summary_rows = []
    for key in model_keys:
        met = results[key]["metrics"]
        row = {
            "model": pretty[key],
            "test_seed": test_seed,
            "split_seed_train_val": results[key]["split_seed"] if use_val else np.nan,
            "alpha": results[key]["alpha"],
            "cindex_test": met["cindex"]["cindex"],
        }

        # point AUCs
        for t, a in zip(met["auc_horizons"]["times"], met["auc_horizons"]["auc"]):
            row[f"auc_t{t:g}y"] = float(a)

        # CI (if available)
        auc_ci = results[key].get("auc_ci", None)
        if auc_ci is not None:
            for t, lo, hi in zip(auc_ci["times"], auc_ci["ci_low"], auc_ci["ci_high"]):
                row[f"auc_t{t:g}y_low"] = float(lo)
                row[f"auc_t{t:g}y_high"] = float(hi)

        summary_rows.append(row)

    pd.DataFrame(summary_rows).to_csv(tables_dir / "metrics_summary_2models.csv", index=False)


if __name__ == "__main__":
    main()
