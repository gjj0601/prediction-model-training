from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import cumulative_dynamic_auc


@dataclass
class CoxnetModelBundle:
    """
    你 run.py 里持有这个对象即可：
      - bundle.pipeline: 预处理 + Coxnet
      - bundle.alpha:    你选定用于预测的 alpha（None 表示用路径最后一个）
      - bundle.feature_names_out: 预处理后的最终特征名（便于后续系数/解释）
    """
    pipeline: Pipeline
    alpha: Optional[float]
    feature_names_out: Optional[np.ndarray] = None




def _build_preprocessor(meta, ordinal_map):
    transformers = []

    ordinal_cols = meta.get("ordinal", [])
    if ordinal_cols:
        categories = []
        for c in ordinal_cols:
            if c not in ordinal_map:
                raise KeyError(f"Ordinal column '{c}' missing in cfg['data']['preprocessing']['ordinal']")
            categories.append(ordinal_map[c])

        # ① 先把缺失填成一个字符串 "missing"（避免 encoder 输出 NaN）
        # ② 再做有序编码（注意：如有缺失，"missing" 不在 categories 会走 unknown -> -1）
        ord_pipe = Pipeline([
            ("imp", SimpleImputer(strategy="constant", fill_value="missing")),  # :contentReference[oaicite:2]{index=2}
            ("enc", OrdinalEncoder(
                categories=categories,
                handle_unknown="use_encoded_value",
                unknown_value=-1,
                dtype=float,
            )),
        ])
        transformers.append(("ord", ord_pipe, ordinal_cols))

    nominal_cols = meta.get("nominal", [])
    if nominal_cols:
        nom_pipe = Pipeline([
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        transformers.append(("nom", nom_pipe, nominal_cols))

    numeric_cols = meta.get("numeric", [])
    if numeric_cols:
        num_pipe = Pipeline([
            ("imp", SimpleImputer(strategy="median")),  # median impute :contentReference[oaicite:5]{index=5}
            ("sc", StandardScaler(with_mean=True, with_std=True)),
        ])
        transformers.append(("num", num_pipe, numeric_cols))

    pre = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.0,
        verbose_feature_names_out=True,
    )
    return pre


def train_coxnet(
    X_train: pd.DataFrame,
    y_train: Any,
    meta: Dict[str, List[str]],
    cfg: Dict[str, Any],
) -> CoxnetModelBundle:
    """
    训练 elastic-net Cox（Coxnet）模型：
    - 参数来自 cfg["train"]["model_params"]，对应 CoxnetSurvivalAnalysis 文档参数。:contentReference[oaicite:6]{index=6}
    """
    ordinal_map = cfg["data"].get("preprocessing", {}).get("ordinal", {})

    pre = _build_preprocessor(meta=meta, ordinal_map=ordinal_map)

    mp = cfg["train"]["model_params"]
    coxnet = CoxnetSurvivalAnalysis(
        n_alphas=mp.get("n_alphas", 100),
        alphas=mp.get("alphas", None),
        alpha_min_ratio=mp.get("alpha_min_ratio", "auto"),
        l1_ratio=mp.get("l1_ratio", 0.5),
        tol=mp.get("tol", 1e-7),
        max_iter=mp.get("max_iter", 100000),
        verbose=mp.get("verbose", False),
        fit_baseline_model=mp.get("fit_baseline_model", False),
        normalize=False,  # 文档建议：要标准化用 StandardScaler。:contentReference[oaicite:7]{index=7}
    )

    pipe = Pipeline([
        ("preprocess", pre),
        ("coxnet", coxnet),
    ])

    pipe.fit(X_train, y_train)

    # 记录最终特征名（便于导出系数、做解释）
    try:
        fn_out = pipe.named_steps["preprocess"].get_feature_names_out()
    except Exception:
        fn_out = None

    # alpha 先不选（默认：predict 用路径最后一个 alpha）:contentReference[oaicite:8]{index=8}
    return CoxnetModelBundle(pipeline=pipe, alpha=None, feature_names_out=fn_out)


def set_alpha(bundle: CoxnetModelBundle, alpha: Optional[float]) -> CoxnetModelBundle:
    """给已有 bundle 指定预测用的 alpha（None=用最后一个 alpha）。:contentReference[oaicite:9]{index=9}"""
    bundle.alpha = alpha
    return bundle


def predict_risk(bundle: CoxnetModelBundle, X: pd.DataFrame) -> np.ndarray:
    """
    输出 risk score（线性预测器）；alpha=None 用路径最后一个。:contentReference[oaicite:10]{index=10}
    """
    pre = bundle.pipeline.named_steps["preprocess"]
    est = bundle.pipeline.named_steps["coxnet"]

    Xt = pre.transform(X)
    return est.predict(Xt, alpha=bundle.alpha)


def pick_alpha_by_auc(
    bundle: CoxnetModelBundle,
    X_train: pd.DataFrame,
    y_train: Any,
    X_val: pd.DataFrame,
    y_val: Any,
    horizons_years: List[float],
) -> Tuple[float, pd.DataFrame]:
    """
    在 Coxnet 的 alpha 路径上选一个“最优 alpha”：
    用 cumulative_dynamic_auc 在给定时间点（5/10/15）算 AUC(t)，取 mean(AUC_t) 最大者。

    cumulative_dynamic_auc 定义：在时间 t 区分 “t 前发生事件(cases)” vs “t 时未发生(controls)”，可处理删失。:contentReference[oaicite:11]{index=11}
    """
    pre = bundle.pipeline.named_steps["preprocess"]
    est = bundle.pipeline.named_steps["coxnet"]
    alphas = np.asarray(est.alphas_, dtype=float)

    Xt_val = pre.transform(X_val)

    rows = []
    for a in alphas:
        risk = est.predict(Xt_val, alpha=float(a))  # 可指定 alpha；否则默认用最后一个。:contentReference[oaicite:12]{index=12}
        auc_t, _mean_auc = cumulative_dynamic_auc(
            survival_train=y_train,
            survival_test=y_val,
            estimate=risk,
            times=np.asarray(horizons_years, dtype=float),
        )
        rows.append({
            "alpha": float(a),
            **{f"auc_{t:g}y": float(v) for t, v in zip(horizons_years, auc_t)},
            "auc_mean": float(np.mean(auc_t)),
        })

    tbl = pd.DataFrame(rows).sort_values("auc_mean", ascending=False).reset_index(drop=True)
    best_alpha = float(tbl.loc[0, "alpha"])
    return best_alpha, tbl

