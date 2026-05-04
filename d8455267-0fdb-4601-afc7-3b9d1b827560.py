# src/visualization_surv.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np


def _savefig(save_path: Path | str | None, dpi: int = 600) -> None:
    """统一保存参数：dpi + bbox_inches='tight'。"""
    if save_path is None:
        return
    import matplotlib.pyplot as plt

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")


def plot_roc_at_time_multi(
    curves: List[dict],
    t_years: float,
    title: str,
    save_path: Path | str | None = None,
    show: bool = False,
    dpi: int = 600,
) -> None:
    """
    单张图：在某个时间点 t 的 time-dependent ROC(t)，画多模型曲线。
    curves: [{"label": str, "fpr": array, "tpr": array, "auc": float(optional)}...]
    """
    import matplotlib.pyplot as plt

    plt.figure()

    for c in curves:
        lab = c["label"]
        if "auc" in c and c["auc"] is not None:
            lab = f"{lab} (AUC={float(c['auc']):.2f})"

        plt.plot(c["fpr"], c["tpr"], lw=2, label=lab)

    plt.plot([0, 1], [0, 1], "--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{title} (t={t_years:g}y)")
    plt.legend(loc="lower right")
    plt.tight_layout()

    _savefig(save_path, dpi=dpi)
    if show:
        plt.show()
    plt.close()


def plot_auc_over_time_multi(
    series_list: List[dict],
    title: str,
    save_path: Path | str | None = None,
    show: bool = False,
    dpi: int = 600,
) -> None:
    """
    单张图：AUC(t) 随时间变化曲线（每条线一个模型）。
    series_list: [{"label": str, "times": array, "auc": array, "cindex": float(optional)}...]
    """
    import matplotlib.pyplot as plt

    plt.figure()

    for s in series_list:
        times = np.asarray(s["times"], float)
        aucs = np.asarray(s["auc"], float)

        lab = s["label"]
        if "cindex" in s and s["cindex"] is not None:
            lab = f"{lab} (C={float(s['cindex']):.2f})"

        plt.plot(times, aucs, lw=2, label=lab)

    plt.xlabel("Time (years)")
    plt.ylabel("Time-dependent AUC(t)")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()

    _savefig(save_path, dpi=dpi)
    if show:
        plt.show()
    plt.close()


def save_four_figures_pan_style(
    out_dir: Path | str,
    roc_curves_by_t: Dict[float, List[dict]],
    auc_over_time_series: List[dict],
    horizons: Sequence[float] = (5, 10, 15),
    roc_title: str = "Time-dependent ROC (Test)",
    auc_title: str = "Time-dependent AUC(t) over time (Test)",
    prefix: str = "",
    show: bool = False,
    dpi: int = 600,
) -> Dict[str, Path]:
    """
    固定输出 4 张图：
      1) ROC@5y（三模型）
      2) ROC@10y（三模型）
      3) ROC@15y（三模型）
      4) AUC(t) over time（三模型）
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved: Dict[str, Path] = {}

    for t in horizons:
        if float(t) not in roc_curves_by_t:
            raise KeyError(f"roc_curves_by_t missing key: {t}")

        p = out_dir / f"{prefix}roc_t{int(t)}y_3models.png"
        plot_roc_at_time_multi(
            curves=roc_curves_by_t[float(t)],
            t_years=float(t),
            title=roc_title,
            save_path=p,
            show=show,
            dpi=dpi,
        )
        saved[f"roc_t{int(t)}y"] = p

    p_auc = out_dir / f"{prefix}auc_over_time_3models.png"
    plot_auc_over_time_multi(
        series_list=auc_over_time_series,
        title=auc_title,
        save_path=p_auc,
        show=show,
        dpi=dpi,
    )
    saved["auc_over_time"] = p_auc

    return saved

