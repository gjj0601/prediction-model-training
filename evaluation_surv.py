from __future__ import annotations

from typing import Any, Dict, Iterable, Optional
import numpy as np

from sksurv.metrics import (
    concordance_index_censored,
    concordance_index_ipcw,
    cumulative_dynamic_auc,
)


def _split_surv_structured(y: Any) -> tuple[np.ndarray, np.ndarray]:
    """
    y: sksurv.util.Surv.from_dataframe() 产生的 structured array
       dtype 有两个字段：第一个 bool(event)，第二个 float(time)。
    """
    names = y.dtype.names
    if names is None or len(names) < 2:
        raise ValueError("y must be a structured array with two fields (event, time).")
    e = np.asarray(y[names[0]], dtype=bool)
    t = np.asarray(y[names[1]], dtype=float)
    return e, t


def cindex_metrics(
    y_true: Any,
    risk_score: np.ndarray,
    *,
    method: str = "harrell",
    y_train_for_ipcw: Optional[Any] = None,
    tau: Optional[float] = None,
    tied_tol: float = 1e-8,
) -> Dict[str, float]:
    """
    输出 C-index（以及配对计数）。
    """
    risk_score = np.asarray(risk_score, dtype=float)

    if method.lower() in ("harrell", "censored"):
        event, time = _split_surv_structured(y_true)
        c, conc, disc, tied_risk, tied_time = concordance_index_censored(
            event_indicator=event,
            event_time=time,
            estimate=risk_score,
            tied_tol=tied_tol,
        )
        return {
            "cindex": float(c),
            "concordant": float(conc),
            "discordant": float(disc),
            "tied_risk": float(tied_risk),
            "tied_time": float(tied_time),
        }

    if method.lower() in ("uno", "ipcw"):
        if y_train_for_ipcw is None:
            raise ValueError("y_train_for_ipcw must be provided when method='ipcw'.")
        c, conc, disc, tied_risk, tied_time = concordance_index_ipcw(
            survival_train=y_train_for_ipcw,
            survival_test=y_true,
            estimate=risk_score,
            tau=tau,
            tied_tol=tied_tol,
        )
        return {
            "cindex": float(c),
            "concordant": float(conc),
            "discordant": float(disc),
            "tied_risk": float(tied_risk),
            "tied_time": float(tied_time),
            "tau": float(tau) if tau is not None else np.nan,
        }

    raise ValueError(f"Unknown method: {method}. Use 'harrell' or 'ipcw'.")


def auc_metrics(
    y_train: Any,
    y_test: Any,
    risk_score_test: np.ndarray,
    times: Iterable[float],
    tied_tol: float = 1e-8,
) -> Dict[str, Any]:
    """
    time-dependent AUC(t)：cumulative cases / dynamic controls，处理右删失。
    返回：auc（各 time），mean_auc（给定 time 范围的均值）
    """
    times = np.asarray(list(times), dtype=float)
    risk_score_test = np.asarray(risk_score_test, dtype=float)

    auc, mean_auc = cumulative_dynamic_auc(
        survival_train=y_train,
        survival_test=y_test,
        estimate=risk_score_test,
        times=times,
        tied_tol=tied_tol,
    )
    return {
        "times": times,
        "auc": np.asarray(auc, dtype=float),
        "mean_auc": float(mean_auc),
    }


def bootstrap_auc_ci(
    y_train: Any,
    y_test: Any,
    risk_score_test: np.ndarray,
    times: Iterable[float],
    *,
    n_boot: int = 1000,
    seed: int = 2026,
    tied_tol: float = 1e-8,
) -> Dict[str, Any]:
    """
    AUC(t) 95% CI（bootstrap），不重训模型：
      1) 点估计 = 原始 test 的 cumulative_dynamic_auc（与你昨天的点估计一致）
      2) CI = 对 test 进行 bootstrap 重采样（有放回），每次重算 AUC(t)，取 2.5/97.5 分位数

    注意：cumulative_dynamic_auc 使用 IPCW，需要 y_train 来估删失分布；
          且 survival_test 的时间范围要落在 survival_train 范围内（你昨天已通过 times 裁剪规避）。:contentReference[oaicite:2]{index=2}
    """
    times = np.asarray(list(times), dtype=float)
    risk_score_test = np.asarray(risk_score_test, dtype=float)

    n = len(risk_score_test)
    if n != len(y_test):
        raise ValueError("Length mismatch: risk_score_test and y_test must have the same length.")

    rng = np.random.default_rng(seed)

    # 点估计（固定不变）
    auc_hat, mean_auc_hat = cumulative_dynamic_auc(
        survival_train=y_train,
        survival_test=y_test,
        estimate=risk_score_test,
        times=times,
        tied_tol=tied_tol,
    )
    auc_hat = np.asarray(auc_hat, dtype=float)

    boot = np.full((n_boot, len(times)), np.nan, dtype=float)

    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)   # bootstrap indices
        y_b = y_test[idx]
        r_b = risk_score_test[idx]
        try:
            auc_b, _ = cumulative_dynamic_auc(
                survival_train=y_train,
                survival_test=y_b,
                estimate=r_b,
                times=times,
                tied_tol=tied_tol,
            )
            boot[b, :] = np.asarray(auc_b, dtype=float)
        except Exception:
            # 极少数 bootstrap 样本可能在某些 t 上 case/control 退化或数值检查失败
            continue

    ci_low = np.nanpercentile(boot, 2.5, axis=0)
    ci_high = np.nanpercentile(boot, 97.5, axis=0)

    valid_per_time = np.sum(np.isfinite(boot), axis=0).astype(int)

    return {
        "times": times,
        "auc": auc_hat,
        "mean_auc": float(mean_auc_hat),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n_boot": int(n_boot),
        "seed": int(seed),
        "valid_boot_per_time": valid_per_time,
    }


def survival_metrics_bundle(
    *,
    y_train: Any,
    y_test: Any,
    risk_test: np.ndarray,
    horizons: Iterable[float],
    auc_time_grid: Optional[Iterable[float]] = None,
    cindex_method: str = "harrell",
    tau_ipcw: Optional[float] = None,
) -> Dict[str, Any]:
    """
    一次性把“PAN风格需要的指标”打包：
      - cindex（test）
      - AUC@horizons（5/10/15）
      - AUC(t) over time（可选：用于画 AUC 曲线）
    """
    out: Dict[str, Any] = {}

    out["cindex"] = cindex_metrics(
        y_true=y_test,
        risk_score=risk_test,
        method=cindex_method,
        y_train_for_ipcw=y_train if cindex_method.lower() in ("ipcw", "uno") else None,
        tau=tau_ipcw,
    )

    out["auc_horizons"] = auc_metrics(
        y_train=y_train,
        y_test=y_test,
        risk_score_test=risk_test,
        times=horizons,
    )

    if auc_time_grid is not None:
        out["auc_curve"] = auc_metrics(
            y_train=y_train,
            y_test=y_test,
            risk_score_test=risk_test,
            times=auc_time_grid,
        )

    return out
